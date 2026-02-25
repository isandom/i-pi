"""SO3LR driver using the JAX-MD interface.

This driver uses SO3LR's ``to_jax_md()`` function which integrates JAX-MD's neighbor list and energy function interface. Multiple PIMD beads are evaluated in parallel
via ``jax.vmap``.

Requires: so3lr, jax_md, jax, ase
"""

import os
import json
import time
import numpy as np
from typing import Any, Callable, NamedTuple

# Configure JAX memory management BEFORE importing JAX
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# Set precision before importing JAX
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import jax
import jax.numpy as jnp



from jax_md import space

from ase.io import read
from ase.units import Bohr, Hartree, eV, Angstrom

__DRIVER_NAME__ = "so3lr_jaxmd"
__DRIVER_CLASS__ = "SO3LR_JAXMD_driver"


class KernelState(NamedTuple):
    """GPU-resident mutable kernel state."""

    nbrs_batched: Any
    nbrs_lr_batched: Any


class KernelResults(NamedTuple):
    """Host-consumable kernel outputs."""

    energies_hartree: jax.Array
    forces_hartree_bohr: jax.Array
    virials_hartree: jax.Array
    mu: jax.Array
    mu2: jax.Array
    h: jax.Array
    h2: jax.Array
    overflow: jax.Array


class _PerBeadOutput(NamedTuple):
    """Single-bead model output before batching/unit conversion."""

    energies_ev: jax.Array
    forces_ev_ang: jax.Array
    virials_ev: jax.Array
    dipole_vec: jax.Array
    hirshfeld: jax.Array


def to_fractional_if_periodic(
    stacked_pos_bohr: jax.Array,
    box_bohr: jax.Array | None,
    bohr_to_ang: float | jax.Array,
    vacuum: bool,
    backend: Any,
):
    """Convert to Angstrom and to fractional coordinates when periodic."""
    stacked_pos_ang = stacked_pos_bohr * bohr_to_ang
    if vacuum:
        return stacked_pos_ang, None

    box_ang = box_bohr * bohr_to_ang
    inv_box = backend.linalg.inv(box_ang)
    stacked_pos_frac = backend.einsum("bni,ij->bnj", stacked_pos_ang, inv_box)
    return stacked_pos_frac, box_ang


def build_fused_kernel(
    *,
    energy_fn: Callable,
    update_sr_batched: Callable,
    update_lr_batched: Callable,
    is_vacuum: bool,
    compute_stress: bool,
    bohr_to_ang: jax.Array,
    ev_to_hartree: jax.Array,
    ev_ang_to_hartree_bohr: jax.Array,
    n_atoms: int,
    dtype: Any,
) -> Callable:
    """Create the fused update+compute kernel with fixed feature flags."""
    empty_virial = jnp.zeros((0,), dtype=dtype)
    empty_vec = jnp.zeros((0,), dtype=dtype)

    def _extract_observables(aux, array_dtype):
        dipole = jnp.ravel(aux.get("dipole_vec", jnp.zeros((0,), dtype=array_dtype)))
        hirshfeld = jnp.ravel(
            aux.get("hirshfeld_ratios", jnp.zeros((0,), dtype=array_dtype))
        )
        dipole = jnp.concatenate([dipole, jnp.zeros((3,), dtype=array_dtype)])[:3]
        hirshfeld = jnp.concatenate(
            [hirshfeld, jnp.zeros((n_atoms,), dtype=array_dtype)]
        )[:n_atoms]
        return dipole, hirshfeld

    def _observable_moments(dipoles, hirshfeld):
        mu_mean = jnp.mean(dipoles, axis=0)
        mu2_mean = jnp.mean(dipoles**2, axis=0)
        h_mean = jnp.mean(hirshfeld, axis=0)
        h2_mean = jnp.mean(hirshfeld**2, axis=0)
        return mu_mean, mu2_mean, h_mean, h2_mean

    def create_energy_force_single(compute_observables: bool):
        def energy_force_single(pos, nbr, nbr_lr, box_arg):
            def neg_energy_with_eps(R, eps):
                kwargs = {"neighbor": nbr, "neighbor_lr": nbr_lr, "box": box_arg}
                dim = R.shape[-1]
                kwargs["perturbation"] = jnp.eye(dim, dtype=R.dtype) + eps
                
                if compute_observables:
                    energy_atoms, aux = energy_fn(R, has_aux=True, **kwargs)
                    return -jnp.sum(energy_atoms), _extract_observables(aux, R.dtype)
                return -energy_fn(R, **kwargs)

            def neg_energy_no_eps(R):
                kwargs = {"neighbor": nbr, "neighbor_lr": nbr_lr, "box": box_arg}
                if compute_observables:
                    energy_atoms, aux = energy_fn(R, has_aux=True, **kwargs)
                    return -jnp.sum(energy_atoms), _extract_observables(aux, R.dtype)
                return -energy_fn(R, **kwargs)

            if compute_stress:
                dim = pos.shape[-1]
                zero_eps = jnp.zeros((dim, dim), dtype=pos.dtype)
                
                if compute_observables:
                    (neg_energy_ev, (dipole_vec, hirshfeld)), (forces_ev_ang, virials_ev) = (
                        jax.value_and_grad(neg_energy_with_eps, argnums=(0, 1), has_aux=True)(pos, zero_eps)
                    )
                else:
                    neg_energy_ev, (forces_ev_ang, virials_ev) = (
                        jax.value_and_grad(neg_energy_with_eps, argnums=(0, 1))(pos, zero_eps)
                    )
                    dipole_vec, hirshfeld = empty_vec, empty_vec
                
                virials_ev = 0.5 * (virials_ev + virials_ev.T)
            else:
                if compute_observables:
                    (neg_energy_ev, (dipole_vec, hirshfeld)), forces_ev_ang = (
                        jax.value_and_grad(neg_energy_no_eps, has_aux=True)(pos)
                    )
                else:
                    neg_energy_ev, forces_ev_ang = jax.value_and_grad(neg_energy_no_eps)(pos)
                    dipole_vec, hirshfeld = empty_vec, empty_vec
                virials_ev = empty_virial

            return _PerBeadOutput(
                energies_ev=-neg_energy_ev,
                forces_ev_ang=forces_ev_ang,
                virials_ev=virials_ev,
                dipole_vec=dipole_vec,
                hirshfeld=hirshfeld,
            )
        return energy_force_single

    vmapped_energy_and_force_fn_obs = jax.vmap(
        create_energy_force_single(True), in_axes=(0, 0, 0, None)
    )
    vmapped_energy_and_force_fn_no_obs = jax.vmap(
        create_energy_force_single(False), in_axes=(0, 0, 0, None)
    )

    def update_and_compute(
        nbrs_batched, nbrs_lr_batched, stacked_pos_bohr, box_bohr, compute_observables: bool
    ) -> tuple[KernelState, KernelResults]:
        stacked_pos, box_ang = to_fractional_if_periodic(
            stacked_pos_bohr, box_bohr, bohr_to_ang, is_vacuum, backend=jnp
        )

        nbrs_batched = update_sr_batched(nbrs_batched, stacked_pos, box_ang)
        nbrs_lr_batched = update_lr_batched(nbrs_lr_batched, stacked_pos, box_ang)

        overflow = jnp.any(nbrs_batched.did_buffer_overflow) | jnp.any(
            nbrs_lr_batched.did_buffer_overflow
        )

        if compute_observables:
            results = vmapped_energy_and_force_fn_obs(
                stacked_pos, nbrs_batched.idx, nbrs_lr_batched.idx, box_ang
            )
            mu, mu2, h, h2 = _observable_moments(results.dipole_vec, results.hirshfeld)
        else:
            results = vmapped_energy_and_force_fn_no_obs(
                stacked_pos, nbrs_batched.idx, nbrs_lr_batched.idx, box_ang
            )
            mu, mu2, h, h2 = empty_vec, empty_vec, empty_vec, empty_vec

        virials_hartree = (
            results.virials_ev * ev_to_hartree if compute_stress else empty_virial
        )

        state = KernelState(
            nbrs_batched=nbrs_batched,
            nbrs_lr_batched=nbrs_lr_batched,
        )
        results = KernelResults(
            energies_hartree=results.energies_ev * ev_to_hartree,
            forces_hartree_bohr=results.forces_ev_ang * ev_ang_to_hartree_bohr,
            virials_hartree=virials_hartree,
            mu=mu,
            mu2=mu2,
            h=h,
            h2=h2,
            overflow=overflow,
        )
        return state, results

    return jax.jit(update_and_compute, static_argnames=["compute_observables"], donate_argnums=(0, 1))


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
        disable_cell_list: If True, force-disable JAX-MD cell lists
            even in periodic mode. This is an expert setting useful for
            sparse large boxes where cell-list bookkeeping can dominate.
            Ignored when ``vacuum=True`` because cell lists are already
            disabled in free-space mode. (default: False)
        dtype: "float32" or "float64" (default: "float32")
        verbose: Print diagnostics every 100 evaluations (default: False)
        output_observables: If True, compute and return bead-averaged
            dipole and Hirshfeld moments in extras:
            ``mu``, ``mu2``, ``h``, ``h2``.
        observable_stride: If output_observables is True, this sets how often
            (in steps) they are actually computed. Should match the stride
            of the trajectory block in the i-PI input XML to save computation.
            (default: 1)
    
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
        disable_cell_list: bool = False,
        verbose: bool = False,
        output_observables: bool = False,
        observable_stride: int = 1,
        **kwargs,
    ):
        buffer_size_multiplier = kwargs.pop("buffer_size_multiplier", None)
        if buffer_size_multiplier is not None:
            buffer_size_multiplier = float(buffer_size_multiplier)
            if buffer_size_multiplier_sr == 1.25:
                buffer_size_multiplier_sr = buffer_size_multiplier
            if buffer_size_multiplier_lr == 1.25:
                buffer_size_multiplier_lr = buffer_size_multiplier

        self.calculate_stress = calculate_stress
        self.vacuum = vacuum
        self.verbose = verbose
        self.output_observables = output_observables
        self.observable_stride = max(1, int(observable_stride))

        # Diagnostics
        self.eval_count = 0
        self._empty_json = json.dumps({})
        self._step_times = []

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
            disable_cell_list=disable_cell_list,
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
        
        result = self._compute_batch([cell], [pos])
        return result[0]

    def _format_extras(self, results: KernelResults) -> str:
        """Format auxiliary outputs into i-PI compatible JSON string."""

        return json.dumps(
            {
                "mu": np.asarray(results.mu, dtype=float).reshape(3).tolist(),
                "mu2": np.asarray(results.mu2, dtype=float).reshape(3).tolist(),
                "h": np.asarray(results.h, dtype=float).reshape(-1).tolist(),
                "h2": np.asarray(results.h2, dtype=float).reshape(-1).tolist(),
            }
        )
    
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
        step_start = None
        
        if self.verbose:
            # Make sure GPU is clear before starting the step-timer
            jax.block_until_ready(jnp.array(0.0))
            step_start = time.perf_counter()
        
        # Pre-allocate zero stresses for NVT
        if not self.calculate_stress:
            if self._zero_stresses is None or self._zero_stresses.shape[0] != n_beads:
                self._zero_stresses = np.zeros((n_beads, 3, 3), dtype=self._dtype)
        
        # Stack positions in Bohr on CPU.
        # i-PI FFDirect passes each bead's pos as (n_atoms, 3).
        pos_arr = np.stack(pos_list).astype(self._dtype, copy=False)
        expected_shape = (n_beads, self._n_atoms, 3)
        if pos_arr.shape != expected_shape:
            raise ValueError(
                f"Expected position batch with shape {expected_shape}, "
                f"got {pos_arr.shape}. Check FFDirect inputs."
            )
        stacked_pos_bohr = pos_arr
        
        if self.vacuum:
            box_bohr = None
        else:
            # All beads share the same box (cell is a classical DOF)
            box_bohr = np.asarray(cell_list[0], dtype=self._dtype)
        
        if self._nbrs_batched is None:
            self._allocate_neighbor_lists(stacked_pos_bohr, box_bohr)
        
        do_observables = False
        if self.output_observables:
            if (self.eval_count - 1) % self.observable_stride == 0:
                do_observables = True
        
        # Fused update + compute (unit conversions are inside the JIT kernel)
        kernel_output = self._update_neighbors_and_compute(
            stacked_pos_bohr, box_bohr, compute_observables=do_observables
        )
        energies = kernel_output.energies_hartree
        forces = kernel_output.forces_hartree_bohr
        virials = kernel_output.virials_hartree if self.calculate_stress else None
        
        if do_observables:
            extras_json = self._format_extras(kernel_output)
        else:
            extras_json = self._empty_json
                
        stresses = virials if virials is not None else self._zero_stresses
        
        results = [
            (
                float(energies[i]),
                forces[i].ravel(),
                stresses[i],
                extras_json,
            )
            for i in range(n_beads)
        ]
        
        if self.verbose:
            # GPU is synchronized by jax.device_get() in _update_neighbors_and_compute.
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
        disable_cell_list: bool,
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
        
        if template is None:
            raise ValueError("Must provide 'template' parameter")
        
        template_atoms = read(template)
        template_atoms.set_pbc(True)
        self._n_atoms = len(template_atoms)
        
        species = jnp.array(
            template_atoms.get_atomic_numbers(), 
            dtype=jnp.int32
        )
        
        cell = np.array(template_atoms.get_cell())
        box = jnp.array(cell, dtype=jdtype)
        
        from so3lr import to_jax_md, So3lrPotential
        
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
        disable_cell_list = self.vacuum or disable_cell_list

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
            update_batched = jax.vmap(lambda n, p, bx: n.update(p), in_axes=(0, 0, None))
        else:
            # box is NOT vmapped (in_axes=None) since all beads share same box in NVT
            update_batched = jax.vmap(lambda n, p, bx: n.update(p, box=bx), in_axes=(0, 0, None))

        self._fused_kernel = build_fused_kernel(
            energy_fn=energy_fn,
            update_sr_batched=update_batched,
            update_lr_batched=update_batched,
            is_vacuum=self.vacuum,
            compute_stress=self.calculate_stress,
            bohr_to_ang=bohr_to_ang,
            ev_to_hartree=ev_to_hartree,
            ev_ang_to_hartree_bohr=ev_ang_to_hartree_bohr,
            n_atoms=self._n_atoms,
            dtype=jdtype,
        )
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] Initialized with {self._n_atoms} atoms")
            print(f"[SO3LR-JAXMD] dtype={dtype_str}")
            print(f"[SO3LR-JAXMD] calculate_stress={self.calculate_stress}")
            print(f"[SO3LR-JAXMD] output_observables={self.output_observables} (stride={self.observable_stride})")
            print(f"[SO3LR-JAXMD] disable_cell_list={disable_cell_list}")
            print(f"[SO3LR-JAXMD] Devices: {jax.devices()}")

    def _update_neighbors_and_compute(
        self,
        stacked_pos_bohr: jax.Array,
        box_bohr: jax.Array | None,
        compute_observables: bool,
    ) -> KernelResults:
        """Run the fused update+compute kernel, handling overflow.
        
        The kernel handles all unit conversions internally:
        Bohr → Å → (model evaluation in eV) → Hartree.
        
        Args:
            stacked_pos_bohr: (n_beads, n_atoms, 3) positions in Bohr
            box_bohr: (3, 3) box in Bohr, or None for vacuum
            compute_observables: Dynamically toggle observables computation 
                (triggers branch swap in xla if changed)
        
        Returns:
            KernelResults with converted energies/forces, optional observables,
            and the overflow flag.
        """
        def dispatch() -> KernelResults:
            state, results = self._fused_kernel(
                self._nbrs_batched, self._nbrs_lr_batched, stacked_pos_bohr, box_bohr, compute_observables
            )
            self._nbrs_batched = state.nbrs_batched
            self._nbrs_lr_batched = state.nbrs_lr_batched
            # Keep host transfer here: downstream float()/ravel() should stay CPU-only.
            return jax.device_get(results)

        kernel_output = dispatch()
        if bool(kernel_output.overflow):
            print("[SO3LR-JAXMD] Neighbor list overflow, reallocating with increased capacity...")
            self._nbrs_batched = None
            self._nbrs_lr_batched = None
            self._allocate_neighbor_lists(stacked_pos_bohr, box_bohr)
            print("[SO3LR-JAXMD] Re-triggering JIT compilation due to new buffer shapes. This may take a minute...")
            kernel_output = dispatch()
            if bool(kernel_output.overflow):
                raise RuntimeError("Neighbor list overflow persists after reallocation")

        return kernel_output

    def _allocate_neighbor_lists(
        self, 
        stacked_pos_bohr: jax.Array,
        box_bohr: jax.Array | None,
    ) -> None:
        """Allocate neighbor lists for all PIMD beads.
        
        JAX-MD's allocate() examines the provided positions, counts the actual
        neighbors within the cutoff, and then scales by capacity_multiplier.
        """
        
        stacked_pos, box_ang = to_fractional_if_periodic(
            stacked_pos_bohr, box_bohr, self._bohr_to_ang, self.vacuum, backend=jnp
        )
            
        n_beads = stacked_pos.shape[0]
        
        # Use first bead's position as reference for allocation
        ref_pos = stacked_pos[0]
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] Allocating neighbor lists for {n_beads} beads...")

        def stack_pytree(pytree, copies):
            return jax.tree.map(lambda x: jnp.stack([x] * copies), pytree)

        box_kwargs = {} if self.vacuum else {"box": jnp.asarray(box_ang)}
        
        # Allocate once, then stack N copies of the dynamic leaves.
        nbrs = self._neighbor_fn.allocate(ref_pos, extra_capacity=0, **box_kwargs)
        nbrs_lr = self._neighbor_fn_lr.allocate(ref_pos, extra_capacity=0, **box_kwargs)

        self._nbrs_batched = stack_pytree(nbrs, n_beads)
        self._nbrs_lr_batched = stack_pytree(nbrs_lr, n_beads)
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] Stacked neighbor lists: "
                  f"idx={self._nbrs_batched.idx.shape}, idx_lr={self._nbrs_lr_batched.idx.shape}")
    
