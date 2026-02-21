"""SO3LR driver using the JAX-MD interface.

This driver uses SO3LR's ``to_jax_md()`` function which integrates JAX-MD's neighbor list and energy function interface. Multiple PIMD beads are evaluated in parallel
via ``jax.vmap``.

Requires: so3lr, jax_md, jax, ase
"""

import os
import json
import time
import numpy as np

# Configure JAX memory management BEFORE importing JAX
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# Set precision before importing JAX
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# DEBUG: Log every JAX compilation to identify retracing issues
# Set to "1" to enable, "0" to disable
#os.environ.setdefault("JAX_LOG_COMPILES", "1")

import jax
import jax.numpy as jnp



from jax_md import space

from ase.io import read
from ase.units import Bohr, Hartree, eV, Angstrom

# Driver metadata for i-PI
__DRIVER_NAME__ = "so3lr_jaxmd"
__DRIVER_CLASS__ = "SO3LR_JAXMD_driver"


class SO3LR_JAXMD_driver:
    """SO3LR driver using the JAX-MD interface.
    
    This driver wraps SO3LR's ``to_jax_md()`` function for use with i-PI's
    FFDirect mechanism. It handles batching over PIMD beads using vmap.
    
    Usage with FFDirect::
    
        <ffdirect name="so3lr_ff" threaded="false">
            <pes>so3lr_jaxmd</pes>
            <batch_size>32</batch_size>
            <parameters>{
                template: system.xyz,
                lr_cutoff: 1000.0,
                vacuum: true,
                disable_cell_list: false,
                calculate_stress: false,
                dtype: float32,
                buffer_size_multiplier: 1.25,
                capacity_multiplier: 1.25,
                verbose: false
                }
            </parameters>
        </ffdirect>
    
    Args:
        template: Path to XYZ file defining the system (atomic numbers,
            initial cell, etc.)
        total_charge: Total system charge (default: 0)
        num_unpaired_electrons: Number of unpaired electrons (default: 0)
        lr_cutoff: Long-range cutoff in Angstrom (default: 12.0)
        capacity_multiplier: Buffer multiplier for neighbor lists
            (default: 1.25)
        buffer_size_multiplier: Convenience alias that sets both
            ``buffer_size_multiplier_sr`` and ``buffer_size_multiplier_lr``
            unless they are explicitly overridden.
        buffer_size_multiplier_sr: Buffer multiplier for short-range cell
            lists (default: 1.25)
        buffer_size_multiplier_lr: Buffer multiplier for long-range cell
            lists (default: 1.25)
        calculate_stress: If True, compute the virial tensor via strain
            differentiation (required for NPT). If False, skip the virial
            computation and return zeros (cheaper for NVT). (default: False)
        vacuum: If True, use free-space (no PBC) displacement and disable
            cell lists. Suitable for gas-phase / isolated molecule
            simulations (default: False)
        force_disable_cell_list: If True, force-disable JAX-MD cell lists
            even in periodic mode. This is an expert setting useful for
            sparse large boxes where cell-list bookkeeping can dominate.
            Ignored when ``vacuum=True`` because cell lists are already
            disabled in free-space mode. (default: False)
        disable_cell_list: Alias for ``force_disable_cell_list`` accepted via
            ``**kwargs`` for i-PI parameter compatibility.
        dtype: "float32" or "float64" (default: "float32")
        verbose: Print diagnostics every 100 evaluations (default: False)
    
    Notes:
        - The short-range cutoff is determined by the SO3LR model and
          cannot be changed here.
        - All extra keyword arguments are forwarded to ``so3lr.to_jax_md()``.
    """
    
    def __init__(
        self,
        template: str = None,
        total_charge: int = 0,
        num_unpaired_electrons: int = 0,
        lr_cutoff: float = 12.0,
        capacity_multiplier: float = 1.25,
        buffer_size_multiplier_sr: float = 1.25,
        buffer_size_multiplier_lr: float = 1.25,
        calculate_stress: bool = False,
        dtype: str = "float32",
        vacuum: bool = False,
        force_disable_cell_list: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        # i-PI-friendly alias: disable_cell_list -> force_disable_cell_list
        disable_cell_list_alias = kwargs.pop("disable_cell_list", None)
        if disable_cell_list_alias is not None:
            disable_cell_list_alias = bool(disable_cell_list_alias)
            if force_disable_cell_list and not disable_cell_list_alias:
                raise ValueError(
                    "Conflicting cell-list settings: "
                    "force_disable_cell_list=True but disable_cell_list=False"
                )
            if (not force_disable_cell_list) and disable_cell_list_alias:
                force_disable_cell_list = True

        # Resolve convenience alias for buffer_size_multiplier
        buffer_size_multiplier = kwargs.pop("buffer_size_multiplier", None)
        if buffer_size_multiplier is not None:
            buffer_size_multiplier = float(buffer_size_multiplier)
            if buffer_size_multiplier_sr == 1.25:
                buffer_size_multiplier_sr = buffer_size_multiplier
            if buffer_size_multiplier_lr == 1.25:
                buffer_size_multiplier_lr = buffer_size_multiplier

        # Runtime flags (read every evaluation step)
        self.calculate_stress = calculate_stress
        self.vacuum = vacuum
        self.force_disable_cell_list = force_disable_cell_list
        self.verbose = verbose

        # Diagnostics
        self.eval_count = 0
        self._empty_json = json.dumps({})

        # Mutable state (updated during simulation)
        self._nbrs_batched = None
        self._nbrs_lr_batched = None
        self._zero_stresses = None

        # Build model and JIT-compiled kernels.
        # Everything else (template_path, lr_cutoff, species, box, etc.)
        # is consumed here and not stored on self.
        self._setup_model_and_kernels(
            template=template,
            total_charge=total_charge,
            num_unpaired_electrons=num_unpaired_electrons,
            lr_cutoff=lr_cutoff,
            capacity_multiplier=capacity_multiplier,
            buffer_size_multiplier_sr=buffer_size_multiplier_sr,
            buffer_size_multiplier_lr=buffer_size_multiplier_lr,
            dtype_str=dtype,
            extra_kwargs=kwargs,
        )

    def __call__(
        self,
        cell: np.ndarray | list[np.ndarray],
        pos: np.ndarray | list[np.ndarray],
    ) -> tuple | list[tuple]:
        """Unified interface for single or batched evaluation.
        
        Handles both:
        - Single structure: cell is (3,3) array, pos is (n_atoms, 3) array
        - Batch of structures: cell and pos are lists (from FFDirect batch_size > 1)
        
        Returns:
            Single: (energy, forces, stress, extras) tuple
            Batch: List of (energy, forces, stress, extras) tuples
        """
        if isinstance(cell, list):
            return self._compute_batch(cell, pos)
        
        # Single structure: wrap as batch of 1, unwrap result
        result = self._compute_batch([cell], [pos])
        return result[0]
    
    def _compute_batch(
        self,
        cell_list: list[np.ndarray],
        pos_list: list[np.ndarray],
    ) -> list[tuple[float, np.ndarray, np.ndarray, str]]:
        """Batched evaluation for PIMD beads using vmap.
        
        Computes energies and forces for all beads in parallel on the GPU
        via jax.vmap, with a fused neighbor-list-update + energy/force kernel.
        
        Args:
            cell_list: List of (3, 3) cell matrices in Bohr
            pos_list: List of (n_atoms, 3) positions in Bohr
        
        Returns:
            List of (energy, forces, stress, extras) tuples
        """
        self.eval_count += 1
        n_beads = len(pos_list)
        
        if self.verbose:
            if not hasattr(self, "_step_times"):
                self._step_times = []
            # Make sure GPU is clear before starting the step-timer
            jax.block_until_ready(jnp.array(0.0))
            step_start = time.perf_counter()
        
        # Pre-allocate zero stresses for NVT (no virial computation)
        if not self.calculate_stress:
            if self._zero_stresses is None or self._zero_stresses.shape[0] != n_beads:
                self._zero_stresses = np.zeros((n_beads, 3, 3), dtype=self._dtype)
        
        # Stack positions in Bohr on CPU
        pos_arr = np.asarray(pos_list, dtype=self._dtype)
        stacked_pos_bohr = jax.device_put(pos_arr.reshape(n_beads, self._n_atoms, 3))
        
        # Box in Bohr (or None for vacuum)
        if self.vacuum:
            box_bohr = None
        else:
            # All beads share the same box (cell is a classical DOF, not replicated)
            box_bohr = jax.device_put(np.asarray(cell_list[0], dtype=self._dtype))
        
        # Ensure neighbor lists are allocated
        # Only runs on first call or after overflow
        if self._nbrs_batched is None:
            self._allocate_neighbor_lists(stacked_pos_bohr, box_bohr)
        
        # Fused update + compute (unit conversions are inside the JIT kernel)
        kernel_results = self._update_neighbors_and_compute(
            stacked_pos_bohr, box_bohr
        )
        energies = kernel_results[0]
        forces = kernel_results[1]
        
        # Virial (already in Hartree from the kernel)
        if self.calculate_stress:
            virials = kernel_results[2]
        
        # Format results
        results = [
            (
                float(energies[i]),
                forces[i].ravel(),
                virials[i] if self.calculate_stress else self._zero_stresses[i],
                self._empty_json,
            )
            for i in range(n_beads)
        ]
        
        if self.verbose:
            # Block until forces are fully computed to capture true GPU time
            for _, f, _, _ in results:
                jax.block_until_ready(f)
            
            step_time = time.perf_counter() - step_start
            self._step_times.append(step_time)
            
            if self.eval_count % 100 == 0:
                times = np.array(self._step_times)
                mean_time = np.mean(times)
                std_time = np.std(times)
                print(f"[SO3LR-JAXMD] Execution stats (last {len(times)} steps): "
                      f"{mean_time:.3f} \u00b1 {std_time:.3f} s/step")
                self._step_times.clear()
        
        return results

    def _setup_model_and_kernels(
        self,
        template: str,
        total_charge: int,
        num_unpaired_electrons: int,
        lr_cutoff: float,
        capacity_multiplier: float,
        buffer_size_multiplier_sr: float,
        buffer_size_multiplier_lr: float,
        dtype_str: str,
        extra_kwargs: dict,
    ) -> None:
        """Initialize the model, build JIT-compiled kernels.
        
        All configuration parameters are consumed here. Only the compiled
        kernels, neighbor-list allocators, and minimal runtime state are
        stored on ``self``.
        """
        
        # Configure JAX precision
        if dtype_str == "float64":
            jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        
        self._dtype = np.float32 if dtype_str == "float32" else np.float64
        jdtype = jnp.float32 if dtype_str == "float32" else jnp.float64

        # Unit conversion scalars (JAX arrays, live on GPU).
        # Captured by the JIT kernel closure so conversions are fused
        # into the compiled program with zero dispatch overhead.
        # i-PI: Bohr, Hartree  ↔  SO3LR/JAX-MD: Angstrom, eV
        self._bohr_to_ang = jnp.array(Bohr / Angstrom, dtype=jdtype)
        bohr_to_ang = self._bohr_to_ang
        ev_to_hartree = jnp.array(eV / Hartree, dtype=jdtype)
        ev_ang_to_hartree_bohr = jnp.array(
            (eV / Hartree) * (Bohr / Angstrom), dtype=jdtype
        )
        
        # Load template
        if template is None:
            raise ValueError("Must provide 'template' parameter")
        
        template_atoms = read(template)
        template_atoms.set_pbc(True)
        self._n_atoms = len(template_atoms)
        
        # Get species
        species = jnp.array(
            template_atoms.get_atomic_numbers(), 
            dtype=jnp.int32
        )
        
        # Get initial box
        cell = np.array(template_atoms.get_cell())
        box = jnp.array(cell, dtype=jdtype)
        
        # Import SO3LR
        from so3lr import to_jax_md, So3lrPotential
        
        # Create potential with configurable lr_cutoff
        potential = So3lrPotential(
            dtype=jdtype,
            lr_cutoff=lr_cutoff,
        )
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] lr_cutoff={lr_cutoff} Å")
        
        # Create displacement function based on vacuum mode
        if self.vacuum:
            # Vacuum/gas-phase: no PBC, no cell lists
            # This avoids O(box_volume) memory scaling for large boxes
            fractional = False
            box = None
            displacement, _ = space.free()
            if self.verbose:
                print("[SO3LR-JAXMD] Vacuum mode: using space.free(), box=None, disable_cell_list=True")
        else:
            # Periodic system: use fractional coordinates for NPT compatibility
            fractional = True
            displacement, _ = space.periodic_general(
                box=box, 
                fractional_coordinates=fractional
            )
        
        # Get JAX-MD interface
        disable_cell_list = self.vacuum or self.force_disable_cell_list

        self._neighbor_fn, self._neighbor_fn_lr, energy_fn = to_jax_md(
            potential=potential,
            displacement_or_metric=displacement,
            box_size=box,
            species=species,
            capacity_multiplier=capacity_multiplier,
            buffer_size_multiplier_sr=buffer_size_multiplier_sr,
            buffer_size_multiplier_lr=buffer_size_multiplier_lr,
            fractional_coordinates=fractional,
            disable_cell_list=disable_cell_list,
            total_charge=float(total_charge),
            num_unpaired_electrons=float(num_unpaired_electrons),
            dtype=jdtype,
            **extra_kwargs,
        )
        
        # JIT-compile batched neighbor list update functions.
        # vmap slices PyTree leaves so each mapped instance sees an unbatched
        # NeighborList. Static fields (update_fn) remain shared.
        if self.vacuum:
            update_sr_batched = jax.vmap(lambda n, p: n.update(p), in_axes=(0, 0))
            update_lr_batched = jax.vmap(lambda n, p: n.update(p), in_axes=(0, 0))
        else:
            # box is NOT vmapped (in_axes=None) since all beads share same box in NVT
            update_sr_batched = jax.vmap(lambda n, p, bx: n.update(p, box=bx), in_axes=(0, 0, None))
            update_lr_batched = jax.vmap(lambda n, p, bx: n.update(p, box=bx), in_axes=(0, 0, None))

        # Pre-create vmapped energy+force (+virial) kernel.
        # Two code paths: with virial (NPT) and without (NVT), so that
        # NVT does not pay the cost of strain differentiation.
        if self.calculate_stress:
            # NPT path: differentiate w.r.t. both positions (→ forces)
            # and a strain perturbation matrix (→ virial).
            # The perturbation kwarg is handled by SO3LR's featurizer:
            # it transforms both positions and box by the perturbation matrix.
            def energy_force_virial_single(pos, nbr, nbr_lr, box_arg):
                eps0 = jnp.eye(3, dtype=pos.dtype)
                def neg_energy(R, perturbation):
                    return -energy_fn(
                        R, neighbor=nbr, neighbor_lr=nbr_lr,
                        box=box_arg, perturbation=perturbation,
                    )
                neg_E, (forces, neg_virial) = jax.value_and_grad(
                    neg_energy, argnums=(0, 1)
                )(pos, eps0)
                return -neg_E, forces, neg_virial

            # box_arg is not vmapped (shared across beads, or None for vacuum)
            vmapped_energy_and_force_fn = jax.vmap(
                energy_force_virial_single, in_axes=(0, 0, 0, None)
            )
        else:
            # NVT path: forces only, no strain differentiation overhead.
            def energy_and_force_single(pos, nbr, nbr_lr, box_arg):
                def neg_energy(R):
                    return -energy_fn(
                        R, neighbor=nbr, neighbor_lr=nbr_lr, box=box_arg,
                    )
                neg_E, forces = jax.value_and_grad(neg_energy)(pos)
                return -neg_E, forces

            # box_arg is not vmapped (shared across beads, or None for vacuum)
            vmapped_energy_and_force_fn = jax.vmap(
                energy_and_force_single, in_axes=(0, 0, 0, None)
            )

        # Build the fused update+compute kernel.
        # All locals captured here (energy_fn, conversion scalars, update
        # functions, vmapped kernel) become part of the closure — they
        # live as long as self._update_and_compute_batched does, but are
        # not individually accessible from outside.
        is_vacuum = self.vacuum
        compute_stress = self.calculate_stress

        def update_and_compute(nbrs_batched, nbrs_lr_batched, stacked_pos_bohr, box_bohr):
            # --- Input unit conversion (fused into kernel) ---
            stacked_pos_ang = stacked_pos_bohr * bohr_to_ang
            if is_vacuum:
                stacked_pos = stacked_pos_ang
                box_ang = None
            else:
                box_ang = box_bohr * bohr_to_ang
                inv_box = jnp.linalg.inv(box_ang)
                stacked_pos = jnp.einsum('bni,ij->bnj', stacked_pos_ang, inv_box)

            # --- Neighbor list update ---
            if is_vacuum:
                nbrs_batched = update_sr_batched(nbrs_batched, stacked_pos)
                nbrs_lr_batched = update_lr_batched(nbrs_lr_batched, stacked_pos)
            else:
                nbrs_batched = update_sr_batched(nbrs_batched, stacked_pos, box_ang)
                nbrs_lr_batched = update_lr_batched(nbrs_lr_batched, stacked_pos, box_ang)

            # --- Energy / force computation ---
            results = vmapped_energy_and_force_fn(
                stacked_pos, nbrs_batched.idx, nbrs_lr_batched.idx, box_ang,
            )
            overflow = jnp.any(nbrs_batched.did_buffer_overflow) | jnp.any(
                nbrs_lr_batched.did_buffer_overflow
            )

            # --- Output unit conversion (fused into kernel) ---
            if compute_stress:
                energies_ev, forces_ev_ang, virials_ev = results
                energies = energies_ev * ev_to_hartree
                forces = forces_ev_ang * ev_ang_to_hartree_bohr
                virials = virials_ev * ev_to_hartree
                return nbrs_batched, nbrs_lr_batched, energies, forces, virials, overflow
            else:
                energies_ev, forces_ev_ang = results
                energies = energies_ev * ev_to_hartree
                forces = forces_ev_ang * ev_ang_to_hartree_bohr
                return nbrs_batched, nbrs_lr_batched, energies, forces, overflow

        self._update_and_compute_batched = jax.jit(update_and_compute, donate_argnums=(0, 1))
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] Initialized with {self._n_atoms} atoms")
            print(f"[SO3LR-JAXMD] dtype={dtype_str}")
            print(f"[SO3LR-JAXMD] calculate_stress={self.calculate_stress}")
            print(f"[SO3LR-JAXMD] disable_cell_list={disable_cell_list}")
            print(f"[SO3LR-JAXMD] Devices: {jax.devices()}")

    def _update_neighbors_and_compute(
        self,
        stacked_pos_bohr: jnp.ndarray,
        box_bohr: jnp.ndarray | None,
    ) -> tuple:
        """Run the fused update+compute kernel, handling overflow.
        
        The kernel handles all unit conversions internally:
        Bohr → Å → (model evaluation in eV) → Hartree.
        
        Args:
            stacked_pos_bohr: (n_beads, n_atoms, 3) positions in Bohr
            box_bohr: (3, 3) box in Bohr, or None for vacuum
        
        Returns:
            If calculate_stress=False:
                (energies_hartree, forces_hartree_bohr)
            If calculate_stress=True:
                (energies_hartree, forces_hartree_bohr, virials_hartree)
        """
        dispatch_results = self._execute_batched_kernel(stacked_pos_bohr, box_bohr)
        overflow = dispatch_results[-1]
        
        if bool(overflow):
            self._reallocate_on_overflow(stacked_pos_bohr, box_bohr)
            dispatch_results = self._execute_batched_kernel(stacked_pos_bohr, box_bohr)
            overflow = dispatch_results[-1]
            if bool(overflow):
                raise RuntimeError("Neighbor list overflow persists after reallocation")
        
        # Return everything except overflow
        return dispatch_results[:-1]
    
    def _execute_batched_kernel(
        self,
        stacked_pos_bohr: jnp.ndarray,
        box_bohr: jnp.ndarray | None,
    ) -> tuple:
        """Single dispatch of the fused kernel.
        
        The kernel accepts Bohr inputs and returns Hartree outputs.
        All unit conversions are fused inside the JIT-compiled program.
        
        Args:
            stacked_pos_bohr: (n_beads, n_atoms, 3) positions in Bohr
            box_bohr: (3, 3) box in Bohr, or None for vacuum
        
        Returns:
            If calculate_stress=False:
                (energies_hartree, forces_hartree_bohr, overflow)
            If calculate_stress=True:
                (energies_hartree, forces_hartree_bohr, virials_hartree, overflow)
        """
        kernel_outputs = self._update_and_compute_batched(
            self._nbrs_batched, self._nbrs_lr_batched, stacked_pos_bohr, box_bohr
        )
        # First two results are always the updated neighbor lists
        self._nbrs_batched = kernel_outputs[0]
        self._nbrs_lr_batched = kernel_outputs[1]
        # Remaining results: (energies, forces, [virials,] overflow) — already in Hartree
        return jax.device_get(kernel_outputs[2:])

    def _allocate_neighbor_lists(
        self, 
        stacked_pos_bohr: jnp.ndarray,  # (n_beads, n_atoms, 3) positions in Bohr
        box_bohr: jnp.ndarray | None,   # (3, 3) box in Bohr, or None for vacuum
    ) -> None:
        """Allocate neighbor lists for all PIMD beads.
        
        JAX-MD's allocate() examines the provided positions, counts the actual
        neighbors within the cutoff, and then scales by capacity_multiplier.
        """
        
        # Convert to Å for allocation (JAX-MD operates in Angstrom)
        stacked_pos_ang = stacked_pos_bohr * self._bohr_to_ang
        if self.vacuum:
            stacked_pos = stacked_pos_ang
            box_ang = None
        else:
            box_ang = box_bohr * self._bohr_to_ang
            inv_box = jnp.linalg.inv(box_ang)
            stacked_pos = jnp.einsum('bni,ij->bnj', stacked_pos_ang, inv_box)
            
        n_beads = stacked_pos.shape[0]
        
        # Use first bead's position as reference for allocation
        ref_pos = stacked_pos[0]
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] Allocating neighbor lists for {n_beads} beads...")

        def stack_pytree(pytree, copies):
            return jax.tree_util.tree_map(lambda x: jnp.stack([x] * copies), pytree)

        box_kwargs = {} if self.vacuum else {"box": box_ang}
        
        # Allocate once, then stack N copies of the dynamic leaves.
        nbrs = self._neighbor_fn.allocate(ref_pos, extra_capacity=0, **box_kwargs)
        nbrs_lr = self._neighbor_fn_lr.allocate(ref_pos, extra_capacity=0, **box_kwargs)

        self._nbrs_batched = stack_pytree(nbrs, n_beads)
        self._nbrs_lr_batched = stack_pytree(nbrs_lr, n_beads)
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] Stacked neighbor lists: "
                  f"idx={self._nbrs_batched.idx.shape}, idx_lr={self._nbrs_lr_batched.idx.shape}")
    
    def _reallocate_on_overflow(
        self, 
        stacked_pos_bohr: jnp.ndarray,  # (n_beads, n_atoms, 3) positions in Bohr
        box_bohr: jnp.ndarray | None,   # (3, 3) box in Bohr, or None for vacuum
    ) -> None:
        """Reallocate neighbor lists after overflow."""
        print("[SO3LR-JAXMD] Neighbor list overflow, reallocating with increased capacity...")
        
        # Force reallocation by clearing the cached batched neighbor lists
        self._nbrs_batched = None
        self._nbrs_lr_batched = None
        
        self._allocate_neighbor_lists(stacked_pos_bohr, box_bohr)
