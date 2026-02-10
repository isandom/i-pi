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


# Unit conversion constants
# i-PI communicates positions/cell in Bohr and expects energy in Hartree and
# forces in Hartree/Bohr. SO3LR/JAX-MD operates in eV and Angstrom.

# We use ASE's constants.
BOHR_TO_ANG = float(Bohr / Angstrom)
EV_TO_HARTREE = float(eV / Hartree)


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
        vacuum: If True, use free-space (no PBC) displacement and disable
            cell lists. Suitable for gas-phase / isolated molecule
            simulations (default: False)
        dtype: "float32" or "float64" (default: "float32")
        verbose: Print diagnostics every 100 evaluations (default: False)
    
    Notes:
        - ``calculate_stress`` is accepted but currently ignored; stress
          output is always zero.
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
        dtype: str = "float32",
        vacuum: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        buffer_size_multiplier = kwargs.pop("buffer_size_multiplier", None)
        if buffer_size_multiplier is not None:
            buffer_size_multiplier = float(buffer_size_multiplier)
            if buffer_size_multiplier_sr == 1.25:
                buffer_size_multiplier_sr = buffer_size_multiplier
            if buffer_size_multiplier_lr == 1.25:
                buffer_size_multiplier_lr = buffer_size_multiplier

        self.template_path = template
        self.total_charge = total_charge
        self.num_unpaired_electrons = num_unpaired_electrons
        self.lr_cutoff = lr_cutoff
        self.capacity_multiplier = capacity_multiplier
        self.buffer_size_multiplier_sr = buffer_size_multiplier_sr
        self.buffer_size_multiplier_lr = buffer_size_multiplier_lr
        self.dtype_str = dtype
        self.vacuum = vacuum
        self.verbose = verbose
        self._extra_kwargs = kwargs
        
        # State
        self._template_atoms = None
        self._n_atoms = 0
        
        # JAX-MD components
        self._neighbor_fn = None
        self._neighbor_fn_lr = None
        self._energy_fn = None

        # Fused kernels (update neighbor lists + compute E/F in one dispatch)
        self._vmapped_energy_and_force_fn = None
        self._update_and_compute_batched = None
        
        # BATCHED neighbor list state (stacked PyTree for vmapped updates)
        # Array leaves have shape (n_beads, ...), static fields are shared
        self._nbrs_batched = None
        self._nbrs_lr_batched = None
        
        # Diagnostics
        self.eval_count = 0
        self._zero_stresses = None
        self._empty_json = json.dumps({})
        
        # Initialize
        self._initialize()

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
            return self.compute_batch(cell, pos)
        
        # Single structure: wrap as batch of 1, unwrap result
        result = self.compute_batch([cell], [pos])
        return result[0]
    
    def compute_batch(
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
        
        diagnostics = self.verbose and (self.eval_count % 100 == 0)
        
        if diagnostics:
            print(f"[SO3LR-JAXMD] Batch #{self.eval_count}: {n_beads} beads (vmapped)")
            start = time.perf_counter()
            t0 = start
        
        # Pre-allocate stresses
        if self._zero_stresses is None or self._zero_stresses.shape[0] != n_beads:
            self._zero_stresses = np.zeros((n_beads, 3, 3), dtype=self.dtype)
        
        # Stack positions on CPU, single device_put
        pos_arr = np.asarray(pos_list, dtype=self.dtype)
        pos_stacked_np = pos_arr.reshape(n_beads, self._n_atoms, 3) * self._bohr_to_ang
        stacked_pos_ang = jax.device_put(pos_stacked_np)
        
        # Convert to fractional coordinates (or keep Cartesian for vacuum)
        if self.vacuum:
            box_ang = None
            stacked_pos_frac = stacked_pos_ang
        else:
            # NVT: all beads share the same box
            box_np = np.asarray(cell_list[0], dtype=self.dtype) * self._bohr_to_ang
            box_ang = jax.device_put(box_np)
            stacked_pos_frac = self._to_fractional_batched(stacked_pos_ang, box_ang)
        
        if diagnostics:
            jax.block_until_ready(stacked_pos_frac)
            t_convert = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # Ensure neighbor lists are allocated
        self._ensure_neighbor_lists(stacked_pos_frac, box_ang)
        
        if diagnostics:
            t0 = time.perf_counter()
        
        # Fused update + compute
        energies_ev, forces_ev_ang = self._run_fused_kernel(stacked_pos_frac, box_ang)
        
        # Convert units: eV -> Hartree, eV/Angstrom -> Hartree/Bohr
        energies = energies_ev * self._ev_to_hartree
        forces = forces_ev_ang * self._ev_to_hartree * self._bohr_to_ang
        
        # Format results
        results = [
            (
                float(energies[i]),
                forces[i].ravel(),
                self._zero_stresses[i],
                self._empty_json,
            )
            for i in range(n_beads)
        ]
        
        if diagnostics:
            t_compute = time.perf_counter() - t0
            elapsed = time.perf_counter() - start
            print(f"[SO3LR-JAXMD] Total: {elapsed:.3f}s ({elapsed/n_beads:.4f}s/bead)")
            print(f"[SO3LR-JAXMD]   - Convert: {t_convert:.3f}s")
            print(f"[SO3LR-JAXMD]   - Compute: {t_compute:.3f}s")
        
        return results

    # ── Initialization ────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Initialize the driver."""
        
        # Configure JAX precision
        if self.dtype_str == "float64":
            jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        
        self.dtype = np.float32 if self.dtype_str == "float32" else np.float64
        self.jdtype = jnp.float32 if self.dtype_str == "float32" else jnp.float64

        # Typed conversion scalars (avoid NumPy upcasting on multiply)
        self._bohr_to_ang = np.asarray(BOHR_TO_ANG, dtype=self.dtype)
        self._ev_to_hartree = np.asarray(EV_TO_HARTREE, dtype=self.dtype)
        
        # Load template
        if self.template_path is None:
            raise ValueError("Must provide 'template' parameter")
        
        self._template_atoms = read(self.template_path)
        self._template_atoms.set_pbc(True)
        self._n_atoms = len(self._template_atoms)
        
        # Get species
        self._species = jnp.array(
            self._template_atoms.get_atomic_numbers(), 
            dtype=jnp.int32
        )
        
        # Get initial box
        cell = np.array(self._template_atoms.get_cell())
        self._box = jnp.array(cell, dtype=self.jdtype)
        
        # Import SO3LR
        from so3lr import to_jax_md, So3lrPotential
        
        # Create potential with configurable lr_cutoff
        potential = So3lrPotential(
            dtype=self.jdtype,
            lr_cutoff=self.lr_cutoff,
        )
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] lr_cutoff={self.lr_cutoff} Å")
        
        # Create displacement function based on vacuum mode
        if self.vacuum:
            # Vacuum/gas-phase: no PBC, no cell lists
            # This avoids O(box_volume) memory scaling for large boxes
            self._fractional = False
            self._box = None
            displacement, _ = space.free()
            if self.verbose:
                print("[SO3LR-JAXMD] Vacuum mode: using space.free(), box=None, disable_cell_list=True")
        else:
            # Periodic system: use fractional coordinates for NPT compatibility
            self._fractional = True
            displacement, _ = space.periodic_general(
                box=self._box, 
                fractional_coordinates=self._fractional
            )
        
        # Get JAX-MD interface
        self._neighbor_fn, self._neighbor_fn_lr, self._energy_fn = to_jax_md(
            potential=potential,
            displacement_or_metric=displacement,
            box_size=self._box,
            species=self._species,
            capacity_multiplier=self.capacity_multiplier,
            buffer_size_multiplier_sr=self.buffer_size_multiplier_sr,
            buffer_size_multiplier_lr=self.buffer_size_multiplier_lr,
            fractional_coordinates=self._fractional,
            disable_cell_list=self.vacuum,
            **self._extra_kwargs,
        )
        
        # JIT-compile batched neighbor list update functions
        # vmap slices PyTree leaves so each mapped instance sees an unbatched
        # NeighborList. Static fields (update_fn) remain shared.
        if self.vacuum:
            self._update_sr_batched = jax.vmap(lambda n, p: n.update(p), in_axes=(0, 0))
            self._update_lr_batched = jax.vmap(lambda n, p: n.update(p), in_axes=(0, 0))
        else:
            # box is NOT vmapped (in_axes=None) since all beads share same box in NVT
            self._update_sr_batched = jax.vmap(lambda n, p, bx: n.update(p, box=bx), in_axes=(0, 0, None))
            self._update_lr_batched = jax.vmap(lambda n, p, bx: n.update(p, box=bx), in_axes=(0, 0, None))


        # Pre-create vmapped energy+force + fused update-and-compute kernel
        energy_fn_captured = self._energy_fn

        def energy_and_force_single(pos, nbr, nbr_lr, box_arg):
            def neg_energy(R):
                return -energy_fn_captured(R, neighbor=nbr, neighbor_lr=nbr_lr, box=box_arg)
            neg_E, forces = jax.value_and_grad(neg_energy)(pos)
            return -neg_E, forces

        # box_arg is not vmapped (shared across beads, or None for vacuum)
        self._vmapped_energy_and_force_fn = jax.vmap(
            energy_and_force_single, in_axes=(0, 0, 0, None)
        )

        def update_and_compute(nbrs_batched, nbrs_lr_batched, stacked_pos, box_arg):
            if self.vacuum:
                nbrs_batched = self._update_sr_batched(nbrs_batched, stacked_pos)
                nbrs_lr_batched = self._update_lr_batched(nbrs_lr_batched, stacked_pos)
            else:
                nbrs_batched = self._update_sr_batched(nbrs_batched, stacked_pos, box_arg)
                nbrs_lr_batched = self._update_lr_batched(nbrs_lr_batched, stacked_pos, box_arg)
            energies_ev, forces_ev_ang = self._vmapped_energy_and_force_fn(
                stacked_pos, nbrs_batched.idx, nbrs_lr_batched.idx, box_arg,
            )
            overflow = jnp.any(nbrs_batched.did_buffer_overflow) | jnp.any(
                nbrs_lr_batched.did_buffer_overflow
            )
            return nbrs_batched, nbrs_lr_batched, energies_ev, forces_ev_ang, overflow

        self._update_and_compute_batched = jax.jit(update_and_compute, donate_argnums=(0, 1))
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] Initialized with {self._n_atoms} atoms")
            print(f"[SO3LR-JAXMD] dtype={self.dtype_str}")
            print(f"[SO3LR-JAXMD] Devices: {jax.devices()}")

    def _run_fused_kernel(
        self,
        stacked_pos_frac: jnp.ndarray,
        box_ang: jnp.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the fused update+compute kernel, handling overflow.
        
        Args:
            stacked_pos_frac: (n_beads, n_atoms, 3) fractional (or Cartesian for vacuum)
            box_ang: (3, 3) box in Angstrom, or None for vacuum
        
        Returns:
            (energies_ev, forces_ev_ang) as NumPy arrays on host
        """
        energies_ev, forces_ev_ang, overflow = self._dispatch(stacked_pos_frac, box_ang)
        
        if bool(overflow):
            self._handle_overflow(stacked_pos_frac, box_ang)
            energies_ev, forces_ev_ang, overflow = self._dispatch(stacked_pos_frac, box_ang)
            if bool(overflow):
                raise RuntimeError("Neighbor list overflow persists after reallocation")
        
        return energies_ev, forces_ev_ang
    
    def _dispatch(
        self,
        stacked_pos_frac: jnp.ndarray,
        box_ang: jnp.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single dispatch of the fused kernel.
        
        Returns:
            (energies_ev, forces_ev_ang, overflow) as NumPy arrays on host
        """
        (
            self._nbrs_batched,
            self._nbrs_lr_batched,
            energies_ev,
            forces_ev_ang,
            overflow,
        ) = self._update_and_compute_batched(
            self._nbrs_batched, self._nbrs_lr_batched, stacked_pos_frac, box_ang
        )
        return jax.device_get((energies_ev, forces_ev_ang, overflow))

    def _ensure_neighbor_lists(
        self, 
        stacked_pos_frac: jnp.ndarray,  # (n_beads, n_atoms, 3) fractional coords
        box_ang: jnp.ndarray,           # (3, 3) box in Angstrom, or None for vacuum
    ) -> bool:
        """Ensure neighbor lists are allocated for all beads.
        
        Allocation strategy:
        - Allocate a NeighborList per bead using a shared reference configuration.
        - Stack the resulting PyTrees so array leaves have shape (n_beads, ...).
        
        We intentionally allocate from the same reference configuration for all
        beads to guarantee identical shapes (required for stacking/vmap).
        This is only done on first use and after rare capacity overflows.
        
        Args:
            stacked_pos_frac: Pre-computed fractional positions (n_beads, n_atoms, 3)
            box_ang: Box matrix in Angstrom (3, 3), or None for vacuum
        
        Returns False (overflow is checked in the fused update+compute kernel).
        """
        n_beads = stacked_pos_frac.shape[0]
        
        # Allocate on first call
        if self._nbrs_batched is None:
            # Use first bead's position as reference for allocation
            ref_pos_frac = stacked_pos_frac[0]
            
            if self.verbose:
                print(f"[SO3LR-JAXMD] Allocating neighbor lists for {n_beads} beads...")

            def stack_pytree(pytree_list):
                return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *pytree_list)

            box_kwargs = {} if self.vacuum else {"box": box_ang}
            nbrs_list = [
                self._neighbor_fn.allocate(ref_pos_frac, extra_capacity=0, **box_kwargs)
                for _ in range(n_beads)
            ]
            nbrs_lr_list = [
                self._neighbor_fn_lr.allocate(ref_pos_frac, extra_capacity=0, **box_kwargs)
                for _ in range(n_beads)
            ]

            self._nbrs_batched = stack_pytree(nbrs_list)
            self._nbrs_lr_batched = stack_pytree(nbrs_lr_list)
            
            if self.verbose:
                print(f"[SO3LR-JAXMD] Stacked neighbor lists: "
                      f"idx={self._nbrs_batched.idx.shape}, idx_lr={self._nbrs_lr_batched.idx.shape}")
        
        return False
    
    def _handle_overflow(
        self, 
        stacked_pos_frac: jnp.ndarray,  # (n_beads, n_atoms, 3) fractional coords
        box_ang: jnp.ndarray | None,    # (3, 3) box in Angstrom, or None for vacuum
    ) -> None:
        """Reallocate neighbor lists after overflow with increased uniform capacity."""
        print("[SO3LR-JAXMD] Neighbor list overflow, reallocating with increased capacity...")
        
        # Force reallocation by clearing the cached batched neighbor lists
        self._nbrs_batched = None
        self._nbrs_lr_batched = None
        
        # Reallocate - _ensure_neighbor_lists will handle uniform allocation
        self._ensure_neighbor_lists(stacked_pos_frac, box_ang)

    @staticmethod
    @jax.jit
    def _to_fractional_batched(pos_ang: jnp.ndarray, box_ang: jnp.ndarray) -> jnp.ndarray:
        """Convert batched Cartesian positions to fractional coordinates.
        
        Args:
            pos_ang: (n_beads, n_atoms, 3) positions in Angstrom
            box_ang: (3, 3) box matrix in Angstrom (shared across beads for NVT)
        
        Returns:
            (n_beads, n_atoms, 3) fractional coordinates
        """
        inv_box = jnp.linalg.inv(box_ang)
        return jnp.einsum('bni,ij->bnj', pos_ang, inv_box)
