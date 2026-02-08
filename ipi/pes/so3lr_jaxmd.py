"""SO3LR driver using the JAX-MD interface.

This driver uses SO3LR's `to_jax_md()` function which provides a clean
integration with JAX-MD's neighbor list and energy function interface.

Requires: so3lr, jax_md, jax
"""

import os
import json
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Configure JAX memory management BEFORE importing JAX
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# Set precision before importing JAX
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# DEBUG: Log every JAX compilation to identify retracing issues
# Set to "1" to enable, "0" to disable
#os.environ.setdefault("JAX_LOG_COMPILES", "1")

import jax
import jax.numpy as jnp
import jax_md
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
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = float(Hartree / eV)
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV


class SO3LR_JAXMD_driver:
    """SO3LR driver using the JAX-MD interface.
    
    This driver wraps SO3LR's `to_jax_md()` function for use with i-PI's
    FFDirect mechanism. It handles batching over PIMD beads using vmap.
    
    Usage with FFDirect:
        <ffdirect name="so3lr" pes="so3lr_jaxmd" batch_size="8">
            <parameters>
                template: system.xyz
                total_charge: 0
            </parameters>
        </ffdirect>
    
    Args:
        template: Path to XYZ file defining the system (atomic numbers, etc.)
        total_charge: Total system charge (default: 0)
        num_unpaired_electrons: Number of unpaired electrons (default: 0)
        cutoff: Short-range cutoff in Angstrom (default: from model)
        lr_cutoff: Long-range cutoff in Angstrom (default: from model)
        capacity_multiplier: Buffer multiplier for neighbor lists (default: 1.25)
        buffer_size_multiplier: Buffer multiplier for cell lists; applied to both
            short-range and long-range lists unless explicitly overridden with
            buffer_size_multiplier_sr / buffer_size_multiplier_lr.
        dtype: "float32" or "float64" (default: "float32")
        verbose: Print diagnostics (default: False)
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
        self.kwargs = kwargs
        
        # State
        self._initialized = False
        self._template_atoms = None
        self._n_atoms = 0
        
        # JAX-MD components
        self._neighbor_fn = None
        self._neighbor_fn_lr = None
        self._energy_fn = None
        self._energy_and_force_fn = None  # Combined for efficiency

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
    
    def _initialize(self):
        """Initialize the driver."""
        if self._initialized:
            return
        
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
            displacement, shift = space.free()
            if self.verbose:
                print("[SO3LR-JAXMD] Vacuum mode: using space.free(), box=None, disable_cell_list=True")
        else:
            # Periodic system: use fractional coordinates for NPT compatibility
            self._fractional = True
            displacement, shift = space.periodic_general(
                box=self._box, 
                fractional_coordinates=self._fractional
            )
        self._displacement = displacement
        self._shift = shift
        
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
            disable_cell_list=self.vacuum,  # Disable for vacuum to avoid memory explosion
        )
        
        # Create combined energy+force function via value_and_grad
        # This computes both in a single forward+backward pass (optimal)
        # Instead of calling energy_fn and force_fn separately (2 forward passes)
        energy_fn_captured = self._energy_fn
        
        def energy_and_force(positions, **kwargs):
            """Compute energy and forces in one pass using value_and_grad."""
            # Negate energy for gradient (forces = -grad(E))
            def neg_energy(R):
                return -energy_fn_captured(R, **kwargs)
            
            neg_E, forces = jax.value_and_grad(neg_energy)(positions)
            return -neg_E, forces  # Return positive energy
        
        self._energy_and_force_fn = energy_and_force
        
        # JIT-compile neighbor list update functions to avoid recompilation every step
        # This is CRITICAL for performance - without JIT, the internal `cond` in update()
        # causes ~0.5s recompilation overhead per neighbor list per step!
        if self.vacuum:
            # Vacuum mode: no box argument
            @jax.jit
            def update_nbrs_sr(nbrs, positions):
                return nbrs.update(positions)
            
            @jax.jit
            def update_nbrs_lr(nbrs, positions):
                return nbrs.update(positions)
            
            # BATCHED UPDATE: vmap slices PyTree leaves so each mapped instance
            # sees an unbatched NeighborList. Static fields (update_fn) remain shared.
            self._update_sr_batched = jax.jit(
                jax.vmap(lambda n, p: n.update(p), in_axes=(0, 0))
            )
            self._update_lr_batched = jax.jit(
                jax.vmap(lambda n, p: n.update(p), in_axes=(0, 0))
            )
        else:
            # Periodic mode: box as argument
            @jax.jit
            def update_nbrs_sr(nbrs, positions, box):
                return nbrs.update(positions, box=box)
            
            @jax.jit
            def update_nbrs_lr(nbrs, positions, box):
                return nbrs.update(positions, box=box)
            
            # BATCHED UPDATE: box is NOT vmapped (in_axes=None) since all beads share same box in NVT
            self._update_sr_batched = jax.jit(
                jax.vmap(lambda n, p, bx: n.update(p, box=bx), in_axes=(0, 0, None))
            )
            self._update_lr_batched = jax.jit(
                jax.vmap(lambda n, p, bx: n.update(p, box=bx), in_axes=(0, 0, None))
            )
        
        self._update_nbrs_sr = update_nbrs_sr
        self._update_nbrs_lr = update_nbrs_lr

        # ================================================================
        # Pre-create vmapped energy+force + fused update-and-compute kernel
        #
        # Why: the previous flow did
        #   (1) vmapped neighbor update + host sync (overflow check)
        #   (2) vmapped energy/force + host sync (device_get)
        #
        # By fusing update+compute we reduce dispatch count and avoid an
        # extra synchronization on the common no-overflow path.
        # ================================================================
        energy_fn_captured = self._energy_fn
        is_vacuum = self.vacuum

        if is_vacuum:
            def energy_and_force_single_vacuum(pos, nbr, nbr_lr):
                def neg_energy(R):
                    return -energy_fn_captured(R, neighbor=nbr, neighbor_lr=nbr_lr, box=None)

                neg_E, forces = jax.value_and_grad(neg_energy)(pos)
                return -neg_E, forces

            self._vmapped_energy_and_force_fn = jax.jit(
                jax.vmap(energy_and_force_single_vacuum, in_axes=(0, 0, 0))
            )

            def update_and_compute(nbrs_batched, nbrs_lr_batched, stacked_pos):
                nbrs_batched = self._update_sr_batched(nbrs_batched, stacked_pos)
                nbrs_lr_batched = self._update_lr_batched(nbrs_lr_batched, stacked_pos)
                energies_ev, forces_ev_ang = self._vmapped_energy_and_force_fn(
                    stacked_pos,
                    nbrs_batched.idx,
                    nbrs_lr_batched.idx,
                )
                overflow = jnp.any(nbrs_batched.did_buffer_overflow) | jnp.any(
                    nbrs_lr_batched.did_buffer_overflow
                )
                return nbrs_batched, nbrs_lr_batched, energies_ev, forces_ev_ang, overflow

            self._update_and_compute_batched = jax.jit(update_and_compute, donate_argnums=(0, 1))
        else:
            def energy_and_force_single_periodic(pos, nbr, nbr_lr, box_arg):
                def neg_energy(R):
                    return -energy_fn_captured(R, neighbor=nbr, neighbor_lr=nbr_lr, box=box_arg)

                neg_E, forces = jax.value_and_grad(neg_energy)(pos)
                return -neg_E, forces

            self._vmapped_energy_and_force_fn = jax.jit(
                jax.vmap(energy_and_force_single_periodic, in_axes=(0, 0, 0, None))
            )

            def update_and_compute(nbrs_batched, nbrs_lr_batched, stacked_pos, box_arg):
                nbrs_batched = self._update_sr_batched(nbrs_batched, stacked_pos, box_arg)
                nbrs_lr_batched = self._update_lr_batched(nbrs_lr_batched, stacked_pos, box_arg)
                energies_ev, forces_ev_ang = self._vmapped_energy_and_force_fn(
                    stacked_pos,
                    nbrs_batched.idx,
                    nbrs_lr_batched.idx,
                    box_arg,
                )
                overflow = jnp.any(nbrs_batched.did_buffer_overflow) | jnp.any(
                    nbrs_lr_batched.did_buffer_overflow
                )
                return nbrs_batched, nbrs_lr_batched, energies_ev, forces_ev_ang, overflow

            self._update_and_compute_batched = jax.jit(update_and_compute, donate_argnums=(0, 1))
        
        self._initialized = True
        
        if self.verbose:
            print(f"[SO3LR-JAXMD] Initialized with {self._n_atoms} atoms")
            print(f"[SO3LR-JAXMD] dtype={self.dtype_str}")
            print(f"[SO3LR-JAXMD] Devices: {jax.devices()}")
    
    def _get_positions(self, positions: np.ndarray, box: np.ndarray) -> jnp.ndarray:
        """Get positions in the appropriate coordinate system.
        
        For periodic systems: convert to fractional coordinates.
        For vacuum: return Cartesian coordinates directly.
        """
        if self.vacuum or box is None:
            return positions  # Cartesian for vacuum
        else:
            inv_box = jnp.linalg.inv(box)
            return jnp.dot(positions, inv_box)  # Fractional for PBC
    
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

            if self.vacuum:
                nbrs_list = [
                    self._neighbor_fn.allocate(ref_pos_frac, extra_capacity=0)
                    for _ in range(n_beads)
                ]
                nbrs_lr_list = [
                    self._neighbor_fn_lr.allocate(ref_pos_frac, extra_capacity=0)
                    for _ in range(n_beads)
                ]
            else:
                nbrs_list = [
                    self._neighbor_fn.allocate(ref_pos_frac, box=box_ang, extra_capacity=0)
                    for _ in range(n_beads)
                ]
                nbrs_lr_list = [
                    self._neighbor_fn_lr.allocate(ref_pos_frac, box=box_ang, extra_capacity=0)
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
        box_ang: jnp.ndarray,           # (3, 3) box in Angstrom, or None for vacuum
    ):
        """Reallocate neighbor lists after overflow with increased uniform capacity."""
        if self.verbose:
            print("[SO3LR-JAXMD] Neighbor list overflow, reallocating with increased capacity...")
        
        # Force reallocation by clearing the cached batched neighbor lists
        self._nbrs_batched = None
        self._nbrs_lr_batched = None
        self._n_beads_cached = 0
        
        # Reallocate - _ensure_neighbor_lists will handle uniform allocation
        self._ensure_neighbor_lists(stacked_pos_frac, box_ang)
    
    def __call__(
        self,
        cell,  # Can be np.ndarray (single) or List[np.ndarray] (batch)
        pos,   # Can be np.ndarray (single) or List[np.ndarray] (batch)
    ):
        """Unified interface for single or batched evaluation.
        
        This method handles both:
        - Single structure: cell is (3,3) array, pos is (n_atoms, 3) array
        - Batch of structures: cell is list of (3,3) arrays, pos is list of (n_atoms, 3) arrays
        
        FFDirect calls this with lists when batch_size > 1.
        
        Returns:
            Single: (energy, forces, stress, extras) tuple
            Batch: List of (energy, forces, stress, extras) tuples
        """
        # Detect if we're in batch mode (FFDirect passes lists when batch_size > 1)
        if isinstance(cell, list):
            # Batch mode - route to compute_batch
            return self.compute_batch(cell, pos)
        
        # Single structure mode - original __call__ logic
        self.eval_count += 1
        
        diagnostics = self.verbose and (self.eval_count % 100 == 0)
        
        if diagnostics:
            start = time.perf_counter()
            t0 = start
        
        pos_ang_np = np.asarray(pos, dtype=self.dtype).reshape(-1, 3) * self._bohr_to_ang
        stacked_pos_ang = jax.device_put(pos_ang_np[None, ...])

        # For vacuum mode, use box=None; otherwise convert cell
        if self.vacuum:
            box_ang = None
            stacked_pos_frac = stacked_pos_ang  # Cartesian for vacuum
        else:
            box_ang_np = np.asarray(cell, dtype=self.dtype) * self._bohr_to_ang
            box_ang = jax.device_put(box_ang_np)
            stacked_pos_frac = self._to_fractional_batched(stacked_pos_ang, box_ang)
        
        if diagnostics:
            jax.block_until_ready(stacked_pos_frac)
            t_convert = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # Ensure neighbor lists
        self._ensure_neighbor_lists(stacked_pos_frac, box_ang)

        if diagnostics:
            # Neighbor list update is now fused into the compute dispatch
            t_nbrs = 0.0
            t0 = time.perf_counter()
        
        # Update neighbor lists + compute energy/forces in a single dispatch
        if self.vacuum:
            (
                self._nbrs_batched,
                self._nbrs_lr_batched,
                energies_ev,
                forces_ev_ang,
                overflow,
            ) = self._update_and_compute_batched(
                self._nbrs_batched,
                self._nbrs_lr_batched,
                stacked_pos_frac,
            )
        else:
            (
                self._nbrs_batched,
                self._nbrs_lr_batched,
                energies_ev,
                forces_ev_ang,
                overflow,
            ) = self._update_and_compute_batched(
                self._nbrs_batched,
                self._nbrs_lr_batched,
                stacked_pos_frac,
                box_ang,
            )

        energies_ev, forces_ev_ang, overflow = jax.device_get((energies_ev, forces_ev_ang, overflow))
        if bool(overflow):
            self._handle_overflow(stacked_pos_frac, box_ang)
            self._ensure_neighbor_lists(stacked_pos_frac, box_ang)
            if self.vacuum:
                (
                    self._nbrs_batched,
                    self._nbrs_lr_batched,
                    energies_ev,
                    forces_ev_ang,
                    overflow,
                ) = self._update_and_compute_batched(
                    self._nbrs_batched,
                    self._nbrs_lr_batched,
                    stacked_pos_frac,
                )
            else:
                (
                    self._nbrs_batched,
                    self._nbrs_lr_batched,
                    energies_ev,
                    forces_ev_ang,
                    overflow,
                ) = self._update_and_compute_batched(
                    self._nbrs_batched,
                    self._nbrs_lr_batched,
                    stacked_pos_frac,
                    box_ang,
                )
            energies_ev, forces_ev_ang, overflow = jax.device_get((energies_ev, forces_ev_ang, overflow))
            if bool(overflow):
                raise RuntimeError("Neighbor list overflow persists after reallocation")
        
        # Convert units: eV -> Hartree, eV/Angstrom -> Hartree/Bohr
        energy = float(energies_ev[0] * self._ev_to_hartree)
        forces = np.asarray(forces_ev_ang[0]) * self._ev_to_hartree * self._bohr_to_ang
        
        if diagnostics:
            t_compute = time.perf_counter() - t0
            elapsed = time.perf_counter() - start
            print(f"[SO3LR-JAXMD] Eval #{self.eval_count}: {elapsed:.4f}s (convert={t_convert:.4f}s, nbrs={t_nbrs:.4f}s, compute={t_compute:.4f}s)")
        
        # Return in i-PI format
        stress = np.zeros((3, 3), dtype=self.dtype)
        return energy, forces.ravel(), stress, self._empty_json
    
    def compute_batch(
        self,
        cell_list: List[np.ndarray],
        pos_list: List[np.ndarray],
    ) -> List[Tuple[float, np.ndarray, np.ndarray, str]]:
        """Batched evaluation for PIMD beads using vmap.
        
        This method uses jax.vmap to compute energies and forces for all
        beads in parallel on the GPU, providing significant speedup over
        sequential computation.
        
        Optimizations applied:
        - Stack all positions on CPU, single device_put (reduces N transfers to 1)
        - Compute fractional coordinates once and reuse
        - Use block_until_ready for accurate timing
        
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
        
        # ==============================================
        # OPTIMIZATION 1: Stack on CPU, single device_put
        # ==============================================
        # Stack all positions on CPU first (avoids N separate jnp.array calls)
        # OPTIMIZATION: Use np.asarray + reshape to avoid Python loop in list comprehension
        # This handles both flat (3N,) and structured (N,3) inputs efficiently
        pos_arr = np.asarray(pos_list, dtype=self.dtype)
        pos_stacked_np = pos_arr.reshape(n_beads, self._n_atoms, 3) * self._bohr_to_ang

        # Single transfer to GPU (already correct dtype)
        stacked_pos_ang = jax.device_put(pos_stacked_np)
        
        # For vacuum mode, use box=None; otherwise convert cell (NVT: single box)
        if self.vacuum:
            box_ang = None
            stacked_pos_frac = stacked_pos_ang  # Cartesian coords for vacuum
        else:
            # NVT: all beads share the same box, use first cell
            box_np = np.asarray(cell_list[0], dtype=self.dtype) * self._bohr_to_ang
            box_ang = jax.device_put(box_np)
            
            # ==============================================
            # OPTIMIZATION 2: Compute fractional coords ONCE
            # ==============================================
            stacked_pos_frac = self._to_fractional_batched(stacked_pos_ang, box_ang)
        
        if diagnostics:
            jax.block_until_ready(stacked_pos_frac)
            t_convert = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # ==============================================
        # Neighbor list update (with overflow handling)
        # ==============================================
        self._ensure_neighbor_lists(stacked_pos_frac, box_ang)

        if diagnostics:
            # Neighbor list update is now fused into the compute dispatch
            t_nbrs = 0.0
            t0 = time.perf_counter()
        
        # ==============================================
        # Compute energy/forces for all beads via vmap
        # ==============================================
        # Update neighbor lists + compute energy/forces in a single dispatch
        if self.vacuum:
            (
                self._nbrs_batched,
                self._nbrs_lr_batched,
                energies_ev,
                forces_ev_ang,
                overflow,
            ) = self._update_and_compute_batched(
                self._nbrs_batched,
                self._nbrs_lr_batched,
                stacked_pos_frac,
            )
        else:
            (
                self._nbrs_batched,
                self._nbrs_lr_batched,
                energies_ev,
                forces_ev_ang,
                overflow,
            ) = self._update_and_compute_batched(
                self._nbrs_batched,
                self._nbrs_lr_batched,
                stacked_pos_frac,
                box_ang,
            )

        energies_ev, forces_ev_ang, overflow = jax.device_get((energies_ev, forces_ev_ang, overflow))
        if bool(overflow):
            self._handle_overflow(stacked_pos_frac, box_ang)
            self._ensure_neighbor_lists(stacked_pos_frac, box_ang)
            if self.vacuum:
                (
                    self._nbrs_batched,
                    self._nbrs_lr_batched,
                    energies_ev,
                    forces_ev_ang,
                    overflow,
                ) = self._update_and_compute_batched(
                    self._nbrs_batched,
                    self._nbrs_lr_batched,
                    stacked_pos_frac,
                )
            else:
                (
                    self._nbrs_batched,
                    self._nbrs_lr_batched,
                    energies_ev,
                    forces_ev_ang,
                    overflow,
                ) = self._update_and_compute_batched(
                    self._nbrs_batched,
                    self._nbrs_lr_batched,
                    stacked_pos_frac,
                    box_ang,
                )
            energies_ev, forces_ev_ang, overflow = jax.device_get((energies_ev, forces_ev_ang, overflow))
            if bool(overflow):
                raise RuntimeError("Neighbor list overflow persists after reallocation")

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
            print(f"[SO3LR-JAXMD]   - NbrList: {t_nbrs:.3f}s")
            print(f"[SO3LR-JAXMD]   - Compute: {t_compute:.3f}s")
        
        return results
