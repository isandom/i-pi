"""Refactored SO3LR driver using vmap-based batching with NATIVE PADDING.

ARCHITECTURE OVERVIEW:
=====================
This driver integrates the SO3LR machine learning force field with i-PI's Path Integral
Molecular Dynamics (PIMD) engine. It uses jax.vmap to parallelize over PIMD beads.

Performance Optimization:
- Uses GLP's quadratic_neighbor_list directly (no mlffCalculatorSparse overhead).
- Calls So3lr model directly with on-device graph preparation.
- Implements NATIVE PADDING (jraph-style) for safe edge handling.
- Loads model weights only ONCE (via So3lr), avoiding duplicate memory usage.
- Uses FUSED JIT: topology + model in single kernel.

NATIVE PADDING MECHANISM:
========================
Instead of using edge_mask (which requires mlff patches), we add a PADDING NODE
with atomic_number=0 at the end of node arrays. Invalid edges are redirected to
this padding node.

Native init_masks() computes: point_mask = (z != 0)
→ Padding node gets point_mask = 0
→ Any contribution from/to padding node is automatically zeroed
→ NO mlff patches required!

This replicates what jraph.dynamically_batch does, but inside the vmap JIT.

COMPATIBILITY:
=============
Works with VANILLA mlff (no patches needed). All the safety comes from the
native point_mask mechanism in mlff/nn/stacknet/stacknet.py.

NEIGHBOR LIST STRATEGY:
======================
Rebuilds neighbor lists EVERY STEP with force_update=True for correctness.

VACUUM MODE:
===========
When vacuum=True is set, PBC is disabled by passing cell=None to GLP's neighbor
list functions. This provides:
- No minimum image convention (Cartesian displacements only)
- No cell offset computation (offsets are always zero)
- No fractional coordinate conversion
- Ideal for isolated molecules/clusters
"""

import os
import json
import time
import pathlib
import numpy as np
from ase import Atoms
from ase.io import read

# Configure JAX memory management BEFORE importing JAX
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


# --- Unit conversion constants ---
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Fields that use LR capacity (explicit list to avoid size-based ambiguity)
_LR_FIELDS = frozenset({'idx_i_lr', 'idx_j_lr', 'cell_offset_lr', 'distance_lr'})

# Per-ATOM fields that should NOT be padded by edge capacity.
_ATOM_FIELDS = frozenset({'reference_positions', 'positions'})


def _round_up_to_bucket(value, bucket_size):
    """Round value up to nearest bucket boundary with 25% buffer."""
    buffered = int(value * 1.25)
    return ((buffered + bucket_size - 1) // bucket_size) * bucket_size


def _stack_leaf_broadcast(x, n_batch):
    """Broadcast array to include batch dimension."""
    if hasattr(x, 'ndim'):
        return jnp.broadcast_to(x, (n_batch,) + x.shape)
    return x


__DRIVER_NAME__ = "so3lr_vmap_main"
__DRIVER_CLASS__ = "SO3LR_driver"


class SO3LR_driver(object):
    """
    Refactored SO3LR driver using vmap-based batching with native padding.
    
    Uses jraph-style padding node (z=0) for safe edge handling.
    Works with vanilla mlff without any patches.
    
    Args:
        template: Path to XYZ file defining the system (atomic numbers, etc.)
        vacuum: If True, disable PBC (gas-phase/isolated molecule mode)
        lr_cutoff: Long-range cutoff in Angstrom (default: 12.0)
        cutoff: Short-range cutoff in Angstrom (default: 4.5)
        dtype: 'float32' or 'float64' (default: 'float32')
        verbose: Print diagnostics (default: False)
    """

    def __init__(self, verbose=False, *args, **kwargs):
        self.verbose = verbose
        self.args = args
        self.kwargs = kwargs

        # Components
        self.so3lr_calc = None  # Direct So3lr model
        
        # Vacuum mode: disable PBC for isolated molecules
        self.vacuum = kwargs.get('vacuum', False)

        # Template and data
        self.template_atoms = None
        self.atomic_numbers = None
        self.n_atoms = 0
        self._system_template = None

        # Neighbor list machinery
        self._nl_initialized = False
        self._neighbor_template = None
        self._neighbor_allocator = None
        self._glp_update_fn = None

        # Vmapped functions
        self._neighbors_in_axes = None
        
        # FUSED JIT: single kernel for NL update + model compute
        # (replaces the old separate _compute_jit)
        self._fused_update_and_compute = None

        # State / Cache
        self._batched_neighbors = None
        self._n_batch_cached = 0
        self._stacked_template_neighbors = None
        self._template_version = None
        
        # Padding tracking
        self._max_neighbor_capacity_seen = 0
        self._max_neighbor_lr_capacity_seen = 0
        
        # Pre-allocated result caches
        self._empty_json = json.dumps({})
        self._zero_stresses = None

        # Diagnostics
        self.eval_count = 0

        # Initialize
        self._initialize()

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize(self):
        """Initialize the SO3LR calculator and parameters."""
        
        # Enable JAX x64 mode if float64 requested
        dtype_str = self.kwargs.get('dtype', 'float32')
        if dtype_str == 'float64':
            os.environ['JAX_ENABLE_X64'] = 'true'
            if self.verbose:
                print("[SO3LR] ⚠️ Enabling JAX x64 mode for float64 precision")
        
        global jax, jnp
        import jax
        import jax.numpy as jnp
        
        # === PRECISION HARDENING ===
        # Disable TF32 at CUDA driver level (must be set before any CUDA ops)
        # TF32 uses only 10 mantissa bits vs 23 for true float32, causing accuracy issues
        os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
        
        # Configure JAX matmul precision to maximum
        # Options: 'default' (TF32 on compatible GPUs), 'float32' (true FP32), 'highest' (max precision)
        matmul_precision = self.kwargs.get('matmul_precision', 'highest')
        if matmul_precision != 'default':
            jax.config.update("jax_default_matmul_precision", matmul_precision)
            if self.verbose:
                print(f"[SO3LR] ✓ Matmul precision set to '{matmul_precision}' (TF32 disabled via NVIDIA_TF32_OVERRIDE=0)")

        from glp import System, atoms_to_system
        from so3lr import So3lr
        import so3lr as so3lr_pkg

        # Store references
        self._atoms_to_system = atoms_to_system
        self._System = System
        
        if self.verbose:
            print(f"[SO3LR] JAX devices: {jax.devices()}")

        # Load template
        template_path = self.kwargs.get('template')
        if not template_path:
            raise ValueError("Must provide 'template' parameter with path to xyz file")

        self.template_atoms = read(template_path)
        # Set PBC based on vacuum mode
        self.template_atoms.set_pbc(not self.vacuum)
        self.atomic_numbers = self.template_atoms.get_atomic_numbers()
        self.n_atoms = len(self.template_atoms)

        if self.verbose:
            pbc_status = "VACUUM (no PBC)" if self.vacuum else "PERIODIC (PBC enabled)"
            print(f"[SO3LR] Loaded template with {self.n_atoms} atoms, mode: {pbc_status}")

        # Parameters
        self.lr_cutoff = float(self.kwargs.get('lr_cutoff', 12.0))
        self.cutoff = float(self.kwargs.get('cutoff', 4.5))
        dtype_str = self.kwargs.get('dtype', 'float32')
        self.dtype = np.float32 if dtype_str == 'float32' else np.float64
        self.calc_stress = False  # Not yet supported
        self.damping = float(self.kwargs.get('dispersion_energy_cutoff_lr_damping', 2.0))
        self.total_charge = int(self.kwargs.get('total_charge', 0))
        self.num_unpaired_electrons = int(self.kwargs.get('num_unpaired_electrons', 0))

        # Neighbor list parameters (GLP only uses capacity_multiplier, not buffer_size_multiplier)
        self.capacity_multiplier = float(self.kwargs.get('capacity_multiplier', 1.25))

        # Model path
        model_path = self.kwargs.get('model_path')
        if model_path:
            params_dir = pathlib.Path(model_path)
        else:
            params_dir = pathlib.Path(so3lr_pkg.__file__).parent / 'params'
            if not params_dir.exists():
                raise ImportError(f"Could not find params at {params_dir}. Provide 'model_path'.")

        # SO3LR Calculator
        self.so3lr_calc = So3lr(
            calculate_forces=True,
            lr_cutoff=self.lr_cutoff
        )

        # Theory level configuration
        self.num_theory_levels = int(self.kwargs.get('num_theory_levels', 16))
        self.theory_level = int(self.kwargs.get('theory_level', 1))
        
        # Pre-compute theory_mask as one-hot float32
        self._theory_mask_const = jnp.eye(self.num_theory_levels, dtype=jnp.float32)[self.theory_level:self.theory_level+1]
        
        if self.verbose:
            print(f"[SO3LR] Using theory_level={self.theory_level}")
            print("[SO3LR] ✓ Initialized (using NATIVE PADDING - no mlff patches needed)")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _pad_neighbor_list(self, neighbors, target_capacity, target_lr_capacity=None):
        """Pad neighbor list arrays to target capacity."""
        current_capacity = neighbors.centers.shape[0] if hasattr(neighbors, 'centers') else 0
        current_lr_capacity = 0
        if hasattr(neighbors, 'idx_i_lr') and neighbors.idx_i_lr is not None:
            current_lr_capacity = neighbors.idx_i_lr.shape[0]

        pad_amount_sr = max(0, target_capacity - current_capacity)
        pad_amount_lr = max(0, target_lr_capacity - current_lr_capacity) if target_lr_capacity else 0

        if pad_amount_sr == 0 and pad_amount_lr == 0:
            if target_lr_capacity and target_lr_capacity > 0:
                if not hasattr(neighbors, 'idx_i_lr') or neighbors.idx_i_lr is None:
                    neighbors = neighbors._replace(
                        idx_i_lr=jnp.full((target_lr_capacity,), self.n_atoms, dtype=jnp.int32),
                        idx_j_lr=jnp.full((target_lr_capacity,), self.n_atoms, dtype=jnp.int32)
                    )
            return neighbors

        def pad_leaf_with_path(path, arr):
            if not hasattr(arr, 'shape') or arr.ndim == 0:
                return arr
            
            field_name = None
            for key in reversed(path):
                if hasattr(key, 'key'):
                    field_name = key.key
                    break
                elif hasattr(key, 'name'):
                    field_name = key.name
                    break
                elif isinstance(key, str):
                    field_name = key
                    break
            
            if field_name in _ATOM_FIELDS:
                return arr
            
            if field_name in _LR_FIELDS:
                if target_lr_capacity is not None and arr.shape[0] < target_lr_capacity:
                    pad_width = [(0, target_lr_capacity - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                    fill_value = self.n_atoms
                    return jnp.pad(arr, pad_width, mode='constant', constant_values=fill_value)
            else:
                if arr.shape[0] < target_capacity:
                    pad_width = [(0, target_capacity - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                    fill_value = self.n_atoms
                    return jnp.pad(arr, pad_width, mode='constant', constant_values=fill_value)
            return arr

        padded = jax.tree_util.tree_map_with_path(pad_leaf_with_path, neighbors)
        
        if target_lr_capacity and target_lr_capacity > 0:
            if not hasattr(padded, 'idx_i_lr') or padded.idx_i_lr is None:
                padded = padded._replace(
                    idx_i_lr=jnp.full((target_lr_capacity,), self.n_atoms, dtype=jnp.int32),
                    idx_j_lr=jnp.full((target_lr_capacity,), self.n_atoms, dtype=jnp.int32)
                )
        
        if hasattr(padded, 'overflow'):
            padded = padded._replace(overflow=jnp.array(False))
        
        if hasattr(padded, 'capacity'):
            padded = padded._replace(capacity=target_capacity)
        
        return padded

    def _unstack_pytree(self, batched_pytree, n):
        """Unstack batched pytree into list of individual pytrees."""
        flat, tree_def = jax.tree_util.tree_flatten(batched_pytree)
        return [
            jax.tree_util.tree_unflatten(tree_def, [
                leaf[i] if (hasattr(leaf, 'ndim') and leaf.ndim > 0) else leaf
                for leaf in flat
            ])
            for i in range(n)
        ]

    def _compute_in_axes_for_neighbors(self, neighbors_batched):
        """Compute in_axes for neighbors pytree."""
        flat, tree_def = jax.tree_util.tree_flatten(neighbors_batched)
        in_axes = [0 if (hasattr(leaf, 'ndim') and leaf.ndim > 0) else None for leaf in flat]
        return jax.tree_util.tree_unflatten(tree_def, in_axes)

    # =========================================================================
    # Compute Pipeline
    # =========================================================================

    def _prepare_system_template(self, pos_ang_b, cell_ang_b):
        """Ensure system template and neighbor allocator are initialized."""
        if self._system_template is None:
            atoms = self.template_atoms.copy()
            atoms.set_positions(pos_ang_b[0], apply_constraint=False)
            
            if self.vacuum:
                # Vacuum mode: no cell, no PBC
                atoms.set_cell(None, scale_atoms=False)
                atoms.set_pbc(False)
            else:
                # Periodic mode: set cell from input
                atoms.set_cell(cell_ang_b[0].T, scale_atoms=False)
                
            self._system_template = self._atoms_to_system(atoms, dtype=self.dtype)

        if not self._nl_initialized:
            from glp.neighborlist import quadratic_neighbor_list
            
            # CRITICAL: Pass cell=None for vacuum mode
            # GLP's displacement() function checks: if cell is None → use Rb - Ra (no PBC)
            cell_init = None if self.vacuum else self._system_template.cell
            
            # skin=0.0: We always use force_update=True, so skin-based caching is disabled
            self._neighbor_allocator, self._glp_update_fn = quadratic_neighbor_list(
                cell=cell_init, cutoff=self.cutoff, skin=0.0,
                capacity_multiplier=self.capacity_multiplier, lr_cutoff=self.lr_cutoff
            )
            positions_init = jnp.array(pos_ang_b[0], dtype=self.dtype)
            self._neighbor_template = self._neighbor_allocator(positions_init)
            
            # Use the ACTUAL capacities that GLP allocated (already includes
            # capacity_multiplier).  The old N²/4 heuristic massively
            # over-allocated LR edges for small molecules and the 1.25×
            # re-inflation of SR double-counted the multiplier.
            if hasattr(self._neighbor_template, 'idx_i_lr') and self._neighbor_template.idx_i_lr is not None:
                actual_lr_capacity = int(self._neighbor_template.idx_i_lr.shape[0])
            else:
                # Fallback only if GLP somehow didn't produce LR fields
                actual_lr_capacity = max(self.n_atoms * self.n_atoms // 4, 32)
            self._max_neighbor_lr_capacity_seen = max(
                self._max_neighbor_lr_capacity_seen,
                actual_lr_capacity
            )
            
            if hasattr(self._neighbor_template, 'centers'):
                actual_sr_capacity = int(self._neighbor_template.centers.shape[0])
                self._max_neighbor_capacity_seen = max(self._max_neighbor_capacity_seen, actual_sr_capacity)
            
            self._neighbor_template = self._pad_neighbor_list(
                self._neighbor_template,
                self._max_neighbor_capacity_seen,
                self._max_neighbor_lr_capacity_seen
            )
            
            self._nl_initialized = True
            
            if self.verbose:
                mode_str = "VACUUM (cell=None)" if self.vacuum else "PERIODIC"
                print(f"[SO3LR] ✓ GLP neighbor list initialized ({mode_str}, SR: {self._max_neighbor_capacity_seen}, LR: {self._max_neighbor_lr_capacity_seen})")

    def _ensure_stacked_template(self, n_batch, diagnostics=False):
        """Ensure stacked neighbor template and in_axes are ready for the batch size."""
        if (self._stacked_template_neighbors is None or 
            self._n_batch_cached != n_batch or
            self._template_version != id(self._neighbor_template)):
            
            self._stacked_template_neighbors = jax.tree_util.tree_map(
                lambda x: _stack_leaf_broadcast(x, n_batch), self._neighbor_template
            )
            self._template_version = id(self._neighbor_template)
            self._neighbors_in_axes = self._compute_in_axes_for_neighbors(self._stacked_template_neighbors)
            # Invalidate fused kernel when template shape changes
            self._fused_update_and_compute = None
            
            if diagnostics:
                print(f"[SO3LR] Created stacked template for batch_size={n_batch}")

    def _handle_overflow(self, positions_batched, cell_tensors_batched, n_batch):
        """Handle neighbor list overflow by reallocating with increased capacity."""
        print(f"[SO3LR] ⚠️ Overflow detected, growing capacity")
        
        # Unstack and reallocate overflowed beads
        neighbors_list = self._unstack_pytree(self._batched_neighbors, n_batch)
        overflow_np = np.asarray(self._batched_neighbors.overflow)
        
        for i in range(n_batch):
            if overflow_np[i]:
                if self.vacuum:
                    neighbors_list[i] = self._neighbor_allocator(
                        positions_batched[i], new_cell=None
                    )
                else:
                    neighbors_list[i] = self._neighbor_allocator(
                        positions_batched[i], new_cell=cell_tensors_batched[i]
                    )
        
        max_sr_cap = max(n.centers.shape[0] for n in neighbors_list)
        lr_caps = [n.idx_i_lr.shape[0] for n in neighbors_list
                   if hasattr(n, 'idx_i_lr') and n.idx_i_lr is not None]
        max_lr_cap = max(lr_caps) if lr_caps else 0
        
        SR_BUCKET = 500
        LR_BUCKET = 1000
        
        old_sr_cap = self._max_neighbor_capacity_seen
        old_lr_cap = self._max_neighbor_lr_capacity_seen
        
        self._max_neighbor_capacity_seen = max(self._max_neighbor_capacity_seen, 
                                                _round_up_to_bucket(max_sr_cap, SR_BUCKET))
        self._max_neighbor_lr_capacity_seen = max(self._max_neighbor_lr_capacity_seen,
                                                   _round_up_to_bucket(max_lr_cap, LR_BUCKET))
        
        if self._max_neighbor_capacity_seen != old_sr_cap or self._max_neighbor_lr_capacity_seen != old_lr_cap:
            self._fused_update_and_compute = None
            self._stacked_template_neighbors = None
            print(f"[SO3LR] Capacity grown: SR={self._max_neighbor_capacity_seen}, LR={self._max_neighbor_lr_capacity_seen}")
        
        for i in range(len(neighbors_list)):
            neighbors_list[i] = self._pad_neighbor_list(
                neighbors_list[i], self._max_neighbor_capacity_seen, self._max_neighbor_lr_capacity_seen
            )
        
        self._neighbor_template = jax.tree_util.tree_map(lambda x: x, neighbors_list[0])
        self._template_version = id(self._neighbor_template)
        self._batched_neighbors = jax.tree.map(lambda *args: jnp.stack(args), *neighbors_list)
        
        self._stacked_template_neighbors = jax.tree_util.tree_map(
            lambda x: _stack_leaf_broadcast(x, n_batch), self._neighbor_template
        )
        self._n_batch_cached = n_batch

    def _create_fused_update_and_compute(self):
        """Create a SINGLE fused JIT kernel: NL update + model compute.
        
        OPTIMIZATIONS vs old two-dispatch approach:
        ============================================
        1. Single JIT dispatch: NL update + padding + model in one XLA graph.
           XLA can optimize across the entire pipeline (no Python round-trip).
        2. Buffer donation: input neighbor lists are donated to avoid realloc.
        3. Constant hoisting: padding arrays (atomic_numbers, node_mask, 
           batch_segments, etc.) are pre-built once, not recreated per bead.
        4. Overflow check happens OUTSIDE this kernel (caller re-dispatches
           on the rare overflow path).
        
        NATIVE PADDING MECHANISM:
        =========================
        Uses a padding node with z=0 at the end of node arrays.
        Invalid edges are redirected to this padding node.
        init_masks() computes point_mask = (z != 0) → padding node auto-zeroed.
        """
        dtype = self.dtype
        so3lr_calc = self.so3lr_calc
        n_atoms = self.n_atoms
        total_charge = self.total_charge
        num_unpaired_electrons = self.num_unpaired_electrons
        vacuum_mode = self.vacuum
        glp_update_fn = self._glp_update_fn
        neighbors_in_axes = self._neighbors_in_axes
        
        # PADDING NODE INDEX: One node after all real atoms
        PADDING_NODE_IDX = n_atoms
        n_total_nodes = n_atoms + 1
        
        # =================================================================
        # PRE-COMPUTE CONSTANT ARRAYS (hoisted out of vmap)
        # These are identical for every bead and never change.
        # =================================================================
        atomic_numbers_with_pad = jnp.concatenate([
            jnp.array(self.atomic_numbers, dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32)  # z=0 → point_mask=0
        ])
        node_mask_const = jnp.concatenate([
            jnp.ones(n_atoms, dtype=dtype),
            jnp.zeros(1, dtype=dtype)
        ])
        batch_segments_const = jnp.concatenate([
            jnp.zeros(n_atoms, dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32)  # Padding node → padding graph
        ])
        hirshfeld_const = jnp.zeros(n_total_nodes, dtype=dtype)
        forces_placeholder_const = jnp.zeros((n_total_nodes, 3), dtype=dtype)
        total_charge_const = jnp.array([total_charge, 0], dtype=jnp.int16)
        num_unpaired_const = jnp.array([num_unpaired_electrons, 0], dtype=jnp.int16)
        theory_mask_const = jnp.tile(self._theory_mask_const, (2, 1))
        graph_mask_const = jnp.array([True, False])
        energy_placeholder_const = jnp.zeros(2, dtype=dtype)
        
        def detect_valid_and_remap(idx_i, idx_j):
            """Detect valid edges and remap invalid ones to PADDING NODE."""
            valid_mask = (
                (idx_i >= 0) & (idx_j >= 0) & 
                (idx_i < n_atoms) & (idx_j < n_atoms) & 
                (idx_i != idx_j)
            )
            safe_idx_i = jnp.where(valid_mask, idx_i, PADDING_NODE_IDX)
            safe_idx_j = jnp.where(valid_mask, idx_j, PADDING_NODE_IDX)
            return valid_mask, safe_idx_i, safe_idx_j
        
        def compute_offsets_pbc(frac_positions, safe_idx_i, safe_idx_j, valid_mask):
            """Compute integer cell offsets for PBC."""
            frac_i = frac_positions[safe_idx_i]
            frac_j = frac_positions[safe_idx_j]
            disp_frac = frac_j - frac_i
            offset = -jnp.round(disp_frac).astype(jnp.int32)
            return jnp.where(valid_mask[:, None], offset, 0)
        
        if vacuum_mode:
            # =====================================================
            # VACUUM MODE: NL update + model in one function
            # =====================================================
            
            # Pre-compute zero offset constants (vacuum → offsets are always zero).
            # Hoisted out of vmap so they are traced once, not per-bead.
            sr_capacity = self._max_neighbor_capacity_seen
            lr_capacity = self._max_neighbor_lr_capacity_seen
            zero_offset_sr_const = jnp.zeros((sr_capacity, 3), dtype=jnp.int32)
            zero_offset_lr_const = jnp.zeros((lr_capacity, 3), dtype=jnp.int32)
            
            # Pre-compute padding row for positions concat
            pad_row_const = jnp.zeros((1, 3), dtype=dtype)
            
            def compute_single_vacuum(neighbors, positions):
                """Per-bead: update NL + pad + run model (vacuum)."""
                # Update neighbor list in-place
                neighbors = glp_update_fn(positions, neighbors, new_cell=None, force_update=True)
                
                # GLP already excludes self-interactions and pads invalid
                # entries with N (= PADDING_NODE_IDX), so no remapping needed.
                safe_idx_i = neighbors.centers
                safe_idx_j = neighbors.others
                safe_idx_i_lr = neighbors.idx_i_lr
                safe_idx_j_lr = neighbors.idx_j_lr
                
                # Pad positions with padding node
                positions_with_pad = jnp.concatenate([positions, pad_row_const])
                
                inputs = {
                    'positions': positions_with_pad,
                    'atomic_numbers': atomic_numbers_with_pad,
                    'cell_per_atom': None,
                    'node_mask': node_mask_const,
                    'hirshfeld_ratios': hirshfeld_const,
                    'forces': forces_placeholder_const,
                    'batch_segments': batch_segments_const,
                    'cell': None,
                    'idx_i': safe_idx_i,
                    'idx_j': safe_idx_j,
                    'cell_offset': zero_offset_sr_const,
                    'idx_i_lr': safe_idx_i_lr,
                    'idx_j_lr': safe_idx_j_lr,
                    'cell_offset_lr': zero_offset_lr_const,
                    'cell_lr': None,
                    'total_charge': total_charge_const,
                    'num_unpaired_electrons': num_unpaired_const,
                    'theory_mask': theory_mask_const,
                    'graph_mask': graph_mask_const,
                    'energy': energy_placeholder_const,
                }
                
                output = so3lr_calc(inputs)
                energy = output['energy'][0]
                forces = output['forces'][:n_atoms]
                
                return energy, forces, neighbors
            
            vmapped_fn = jax.vmap(
                compute_single_vacuum, in_axes=(neighbors_in_axes, 0)
            )
            
            # Unit conversion scalars — applied OUTSIDE vmap as a single
            # batched multiply, not traced per-bead.
            ev_to_hartree = jnp.array(EV_TO_HARTREE, dtype=dtype)
            ev_ang_to_hartree_bohr = jnp.array(EV_TO_HARTREE * BOHR_TO_ANG, dtype=dtype)
            
            def fused_kernel(template_nbrs, positions):
                energies, forces, updated_nbrs = vmapped_fn(template_nbrs, positions)
                overflow = jnp.any(updated_nbrs.overflow)
                return energies * ev_to_hartree, forces * ev_ang_to_hartree_bohr, updated_nbrs, overflow
            
            return jax.jit(fused_kernel, donate_argnums=(0,))
        
        else:
            # =====================================================
            # PERIODIC MODE: NL update + model in one function
            # =====================================================
            def compute_single(neighbors, positions, cell):
                """Per-bead: update NL + pad + run model (periodic)."""
                # Update neighbor list in-place
                neighbors = glp_update_fn(positions, neighbors, new_cell=cell, force_update=True)
                
                # Topology
                idx_i = neighbors.centers
                idx_j = neighbors.others
                valid_sr, safe_idx_i, safe_idx_j = detect_valid_and_remap(idx_i, idx_j)
                
                idx_i_lr = getattr(neighbors, 'idx_i_lr', None)
                idx_j_lr = getattr(neighbors, 'idx_j_lr', None)
                if idx_i_lr is None:
                    idx_i_lr = jnp.zeros((0,), dtype=jnp.int32)
                if idx_j_lr is None:
                    idx_j_lr = jnp.zeros((0,), dtype=jnp.int32)
                valid_lr, safe_idx_i_lr, safe_idx_j_lr = detect_valid_and_remap(idx_i_lr, idx_j_lr)
                
                # Only positions need padding per-bead
                positions_with_pad = jnp.concatenate([
                    positions,
                    jnp.zeros((1, 3), dtype=dtype)
                ])
                
                # Cell-dependent arrays
                cell_per_atom = jnp.broadcast_to(cell[None, :, :], (n_total_nodes, 3, 3))
                
                n_edges = safe_idx_i.shape[0]
                n_edges_lr = safe_idx_i_lr.shape[0]
                
                inv_cell = jnp.linalg.inv(cell)
                frac_positions = jnp.dot(positions_with_pad, inv_cell)
                final_offset = compute_offsets_pbc(frac_positions, safe_idx_i, safe_idx_j, valid_sr)
                final_offset_lr = compute_offsets_pbc(frac_positions, safe_idx_i_lr, safe_idx_j_lr, valid_lr)
                
                inputs = {
                    'positions': positions_with_pad,
                    'atomic_numbers': atomic_numbers_with_pad,
                    'cell_per_atom': cell_per_atom,
                    'node_mask': node_mask_const,
                    'hirshfeld_ratios': hirshfeld_const,
                    'forces': forces_placeholder_const,
                    'batch_segments': batch_segments_const,
                    'cell': jnp.broadcast_to(cell[None, :, :], (n_edges, 3, 3)),
                    'idx_i': safe_idx_i,
                    'idx_j': safe_idx_j,
                    'cell_offset': final_offset,
                    'idx_i_lr': safe_idx_i_lr,
                    'idx_j_lr': safe_idx_j_lr,
                    'cell_offset_lr': final_offset_lr,
                    'cell_lr': jnp.broadcast_to(cell[None, :, :], (n_edges_lr, 3, 3)),
                    'total_charge': total_charge_const,
                    'num_unpaired_electrons': num_unpaired_const,
                    'theory_mask': theory_mask_const,
                    'graph_mask': graph_mask_const,
                    'energy': energy_placeholder_const,
                }
                
                output = so3lr_calc(inputs)
                energy = output['energy'][0]
                forces = output['forces'][:n_atoms]
                
                energy_hartree = energy * EV_TO_HARTREE
                forces_hartree_bohr = forces * (EV_TO_HARTREE * BOHR_TO_ANG)
                
                return energy_hartree, forces_hartree_bohr, neighbors
            
            vmapped_fn = jax.vmap(
                compute_single, in_axes=(neighbors_in_axes, 0, 0)
            )
            
            def fused_kernel(template_nbrs, positions, cells):
                energies, forces, updated_nbrs = vmapped_fn(template_nbrs, positions, cells)
                overflow = jnp.any(updated_nbrs.overflow)
                return energies, forces, updated_nbrs, overflow
            
            return jax.jit(fused_kernel, donate_argnums=(0,))

    def _dispatch_fused(self, positions_batched, cell_tensors_batched):
        """Single dispatch of the fused NL-update + compute kernel.
        
        Returns:
            (energies, forces, overflow_flag)
            Also updates self._stacked_template_neighbors with the new NL state.
        """
        if self._fused_update_and_compute is None:
            self._fused_update_and_compute = self._create_fused_update_and_compute()
        
        if self.vacuum:
            energies, forces, updated_nbrs, overflow = self._fused_update_and_compute(
                self._stacked_template_neighbors,
                positions_batched,
            )
        else:
            energies, forces, updated_nbrs, overflow = self._fused_update_and_compute(
                self._stacked_template_neighbors,
                positions_batched,
                cell_tensors_batched,
            )
        
        # Store updated neighbors (donated input is now invalid, use output)
        self._stacked_template_neighbors = updated_nbrs
        self._batched_neighbors = updated_nbrs
        
        return energies, forces, overflow

    # =========================================================================
    # Main Interface
    # =========================================================================

    def compute_batch(self, cell_list, pos_list):
        """Main batch compute function using fused NL-update + model kernel.
        
        Single JIT dispatch per step (NL update + padding + model compute).
        Overflow is handled outside the kernel on the rare path.
        """
        self.eval_count += 1
        n_batch = len(pos_list)
        diagnostics = self.verbose and (self.eval_count <= 3 or self.eval_count % 50 == 0)

        if diagnostics:
            print(f"[SO3LR] Batch #{self.eval_count}: {n_batch} structures")
            start = time.time()
            t0 = start
        
        # Pre-allocate zero stresses
        if self._zero_stresses is None or self._zero_stresses.shape[0] != n_batch:
            self._zero_stresses = np.zeros((n_batch, 3, 3), dtype=self.dtype)

        # Unit Conversion (CPU)
        pos_ang_b = (np.asarray(pos_list, dtype=np.float64) * BOHR_TO_ANG).astype(self.dtype)
        positions_batched = jax.device_put(pos_ang_b)

        if self.vacuum:
            # Vacuum: no cell needed — skip conversion + device_put entirely
            cell_ang_b = pos_ang_b  # only used for _prepare_system_template shape
            cell_tensors_batched = None
        else:
            cell_ang_b = (np.asarray(cell_list, dtype=np.float64) * BOHR_TO_ANG).astype(self.dtype)
            cell_tensors_batched = jax.device_put(np.transpose(cell_ang_b, (0, 2, 1)))

        # Initialize system/NL machinery (first call only)
        self._prepare_system_template(pos_ang_b, cell_ang_b)
        
        # Ensure stacked template neighbors are ready
        self._ensure_stacked_template(n_batch, diagnostics)

        if diagnostics:
            t_convert = time.time() - t0
            t0 = time.time()

        # === FUSED DISPATCH: single JIT call does NL update + model compute ===
        energies, forces, overflow = self._dispatch_fused(
            positions_batched, cell_tensors_batched
        )
        
        # Handle overflow (rare path — re-dispatch after reallocation)
        if bool(jax.device_get(overflow)):
            self._handle_overflow(positions_batched, cell_tensors_batched, n_batch)
            # Re-ensure template is valid after overflow reallocation
            self._ensure_stacked_template(n_batch, diagnostics)
            # Second dispatch with grown capacity
            energies, forces, overflow = self._dispatch_fused(
                positions_batched, cell_tensors_batched
            )
            if bool(jax.device_get(overflow)):
                raise RuntimeError("Neighbor list overflow persists after reallocation")
        
        # Transfer results to CPU
        energies_k, forces_k = jax.device_get((energies, forces))
        
        results = [
            (float(energies_k[i]), forces_k[i].ravel(), self._zero_stresses[i], self._empty_json)
            for i in range(n_batch)
        ]

        if diagnostics:
            t_compute = time.time() - t0
            t_total = time.time() - start
            print(f"[SO3LR] Total: {t_total:.3f}s ({t_total/n_batch:.4f}s/struct)")
            print(f"[SO3LR]   - Convert: {t_convert:.3f}s")
            print(f"[SO3LR]   - Fused compute: {t_compute:.3f}s")

        self._n_batch_cached = n_batch
        return results

    def compute_structure(self, cell, pos):
        """Single structure evaluation."""
        return self.compute_batch([cell], [pos])[0]

    def compute(self, cell, pos):
        """Main dispatch."""
        pos_arr = np.asarray(pos)
        if pos_arr.ndim == 3:
            return self.compute_batch(cell, pos)
        return self.compute_structure(cell, pos)

    def __call__(self, cell, pos):
        """Function interface required by i-pi."""
        return self.compute(cell, pos)
