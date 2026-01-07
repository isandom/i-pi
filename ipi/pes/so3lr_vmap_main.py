"""Refactored SO3LR driver using vmap-based batching with Direct Optimization.

ARCHITECTURE OVERVIEW:
=====================
This driver integrates the SO3LR machine learning force field with i-PI's Path Integral
Molecular Dynamics (PIMD) engine. It uses jax.vmap to parallelize over PIMD beads.

Performance Optimization:
- Uses GLP's quadratic_neighbor_list directly (no mlffCalculatorSparse overhead).
- Calls So3lr model directly with on-device graph preparation.
- Implements "Robust Fast Path" offset calculation on GPU (same as jraph_main).
- Loads model weights only ONCE (via So3lr), avoiding duplicate memory usage.

Unit conventions:
- i-PI provides positions and cell in Bohr (atomic units).
- ASE Atoms expects Angstrom.
- So3lr model returns energy in eV, forces in eV/Angstrom.

This driver converts:
- Input Bohr -> Angstrom before constructing systems
- Output eV -> Hartree, eV/Angstrom -> Hartree/Bohr

NOTE ON PADDING:
This driver requires a patched 'mlff' library to correctly handle 'edge_mask'.
Specifically, mlff/nn/embed/embed_sparse.py (GeometryEmbedSparse) must be 
updated to multiply 'cut' by 'edge_mask' if provided. Without this patch,
dummy edges in padded batches will contribute "ghost forces" to the atoms.

NEIGHBOR LIST STRATEGY (Jan 2026 - SIMPLIFIED):
===============================================
This driver rebuilds neighbor lists EVERY STEP using GLP's update_fn with force_update=True.

Why we abandoned skin-based optimization:
- GLP's update_fn (without force_update) only recomputes distances for EXISTING pairs
- It does NOT detect new pairs that come into interaction range as atoms move
- This caused ~1 meV/step energy errors and 20+ meV/ps drift in NVE simulations
- The official SO3LR jax_md example uses dr_threshold=0 (rebuild every step)

Performance impact is minimal:
- GLP update_fn with force_update=True: ~0.6 ms per step (JIT-compiled)
- This is <5% overhead compared to model evaluation (~15 ms)
- Much simpler and more robust than trying to detect topology changes

The skin parameter is kept for GLP's internal buffer sizing but no longer
controls rebuild frequency.
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

# JAX x64 mode: Enable float64 support for numerical stability with large lr_cutoff.
# Large lr_cutoff (e.g., 80 Å) causes float32 overflow in dispersion gradient (R^-11 term).
# This is configured via environment variable so it takes effect before JAX import.
# Usage: set dtype='float64' in driver options to activate.
# NOTE: This env var is checked at JAX import time, so must be set early.
os.environ.setdefault('JAX_ENABLE_X64', 'false')  # Default: off for performance

from ipi.utils import units

# Constants
BOHR_TO_ANG = units.unit_to_user("length", "angstrom", 1.0)
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = units.unit_to_user("energy", "electronvolt", 1.0)
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

# Fields that use LR capacity (explicit list to avoid size-based ambiguity)
_LR_FIELDS = frozenset({'idx_i_lr', 'idx_j_lr', 'cell_offset_lr', 'distance_lr'})

# Per-ATOM fields that should NOT be padded by edge capacity.
# These arrays have shape (n_atoms, ...) not (n_edges, ...).
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
    Refactored SO3LR driver using direct vmap-based batching.
    
    Optimized for PIMD by bypassing intermediate wrappers and computing offsets on GPU.
    """

    def __init__(self, verbose=False, *args, **kwargs):
        self.verbose = verbose
        self.args = args
        self.kwargs = kwargs

        # Components
        self.so3lr_calc = None  # Direct So3lr model

        # Template and data
        self.template_atoms = None
        self.atomic_numbers = None
        self.n_atoms = 0
        self._system_template = None

        # Neighbor list machinery (GLP direct - no mlffCalculatorSparse needed)
        self._nl_initialized = False
        self._neighbor_template = None
        self._neighbor_allocator = None
        self._glp_update_fn = None

        # Vmapped functions
        self._neighbors_in_axes = None
        self._system_in_axes = None
        
        # GPU JIT Functions (optimization)
        self._topology_jit = None       # Computes static inputs (indices, masks) - runs on rebuild only
        self._model_jit = None          # Model execution (uses cached static inputs) - runs every step
        
        # Static Input Cache (optimization: avoid recomputing indices/masks every step)
        self._cached_static_inputs = None  # Dict with safe_idx_i, safe_idx_j, edge_mask, etc.

        # State / Cache
        self._batched_neighbors = None
        self._n_batch_cached = 0
        self._stacked_template_neighbors = None  # Cached stacked template for NL update
        self._template_version = None  # Track when template changes (via id())
        
        # Fast Path cache
        self._cached_batched_system = None  # Cached batched System (static fields reused, dynamic fields overwritten)
        
        # Padding tracking
        # NOTE: LR capacity is initialized to a reasonable default in _prepare_system_template
        # to prevent JIT retraces when molecules that start far apart later come within LR cutoff.
        self._max_neighbor_capacity_seen = 0
        self._max_neighbor_lr_capacity_seen = 0  # Will be set to n_atoms * reasonable_factor
        
        # Pre-allocated result caches (avoid allocation in hot path)
        self._empty_json = json.dumps({})
        self._zero_stresses = None  # Will be initialized with correct batch size

        # Diagnostics
        self.eval_count = 0

        # Initialize
        self._initialize()

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize(self):
        """Initialize the SO3LR calculator and parameters.
        
        OPTIMIZATION: We use GLP's quadratic_neighbor_list directly, bypassing
        mlffCalculatorSparse entirely. This saves:
        - ~1-4 seconds of setup time (no redundant weight loading)
        - ~120-570 MB of GPU memory (weights loaded only once via So3lr)
        """
        # CRITICAL: Enable JAX x64 mode BEFORE importing JAX if float64 requested.
        # This prevents overflow in dispersion gradient (R^-11 term) with large lr_cutoff.
        dtype_str = self.kwargs.get('dtype', 'float32')
        if dtype_str == 'float64':
            os.environ['JAX_ENABLE_X64'] = 'true'
            if self.verbose:
                print("[SO3LR] ⚠️ Enabling JAX x64 mode for float64 precision (required for large lr_cutoff)")
        
        global jax, jnp
        import jax
        import jax.numpy as jnp

        from glp import System, atoms_to_system
        from so3lr import So3lr
        import so3lr as so3lr_pkg

        # Store references for use in _prepare_system_template
        self._atoms_to_system = atoms_to_system
        self._System = System
        
        if self.verbose:
            print(f"[SO3LR] JAX devices: {jax.devices()}")

        # Load template
        template_path = self.kwargs.get('template')
        if not template_path:
            raise ValueError("Must provide 'template' parameter with path to xyz file")

        self.template_atoms = read(template_path)
        self.template_atoms.set_pbc(True)
        self.atomic_numbers = self.template_atoms.get_atomic_numbers()
        self.n_atoms = len(self.template_atoms)

        if self.verbose:
            print(f"[SO3LR] Loaded template with {self.n_atoms} atoms")

        # Parameters
        self.lr_cutoff = float(self.kwargs.get('lr_cutoff', 12.0))
        self.cutoff = float(self.kwargs.get('cutoff', 4.5))
        dtype_str = self.kwargs.get('dtype', 'float32')
        self.dtype = np.float32 if dtype_str == 'float32' else np.float64
        # NOTE: Stress calculation is NOT yet supported by the So3lr wrapper.
        # The model only computes energy and forces. NPT simulations will receive zeros.
        # See PROJECT_CONTEXT.md Section 9 for details and future implementation plans.
        self.calc_stress = False  # Disabled until So3lr supports strain-gradient method
        self.damping = float(self.kwargs.get('dispersion_energy_cutoff_lr_damping', 2.0))
        # FIX: Use int16 dtype for total_charge/num_unpaired_electrons (matches mlff dataloader)
        self.total_charge = int(self.kwargs.get('total_charge', 0))
        self.num_unpaired_electrons = int(self.kwargs.get('num_unpaired_electrons', 0))

        # Neighbor list parameters
        self.skin = float(self.kwargs.get('skin', 1.0))
        self.capacity_multiplier = float(self.kwargs.get('capacity_multiplier', 1.25))
        self.buffer_size_multiplier = float(self.kwargs.get('buffer_size_multiplier', 1.25))

        # Model path (used only by So3lr, not for neighbor list)
        model_path = self.kwargs.get('model_path')
        if model_path:
            params_dir = pathlib.Path(model_path)
        else:
            params_dir = pathlib.Path(so3lr_pkg.__file__).parent / 'params'
            if not params_dir.exists():
                raise ImportError(f"Could not find params at {params_dir}. Provide 'model_path'.")

        # SO3LR Calculator - Used for actual energy/force calculation
        # This is the ONLY place model weights are loaded (optimized from loading twice)
        self.so3lr_calc = So3lr(
            calculate_forces=True,
            lr_cutoff=self.lr_cutoff
        )

        # Theory level configuration (for multi-theory-level models)
        # num_theory_levels: Total number of theory levels (default 16 for SO3LR)
        # theory_level: Which theory level to use (default 1, matching jraph drivers)
        self.num_theory_levels = int(self.kwargs.get('num_theory_levels', 16))
        self.theory_level = int(self.kwargs.get('theory_level', 1))
        
        # FIX: Pre-compute theory_mask as ONE-HOT float32 (not ones int32!)
        # The mlff model uses theory_mask to SELECT per-theory-level parameters.
        # Using jnp.ones would incorrectly sum across ALL theory levels.
        # One-hot encoding: shape (1, num_theory_levels), dtype float32
        self._theory_mask_const = jnp.eye(self.num_theory_levels, dtype=jnp.float32)[self.theory_level:self.theory_level+1]
        
        if self.verbose:
            print(f"[SO3LR] Using theory_level={self.theory_level}, num_theory_levels={self.num_theory_levels}")

        if self.verbose:
            print("[SO3LR] ✓ Initialized So3lr (GLP neighbor lists will be set up on first compute)")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _pad_neighbor_list(self, neighbors, target_capacity, target_lr_capacity=None):
        """Pad neighbor list arrays to target capacity to prevent JIT retraces.
        
        Uses FIELD-NAME-BASED padding to correctly distinguish SR vs LR arrays,
        avoiding the subtle bug where size-based inference could incorrectly pad
        SR arrays with LR capacity when both capacities coincide.
        
        Also ensures idx_i_lr/idx_j_lr exist with proper capacity, even if originally
        missing or None. This is critical for the "distant molecules coming together"
        scenario where LR neighbors may appear after simulation start.
        """
        current_capacity = neighbors.centers.shape[0] if hasattr(neighbors, 'centers') else 0
        current_lr_capacity = 0
        if hasattr(neighbors, 'idx_i_lr') and neighbors.idx_i_lr is not None:
            current_lr_capacity = neighbors.idx_i_lr.shape[0]

        pad_amount_sr = max(0, target_capacity - current_capacity)
        pad_amount_lr = max(0, target_lr_capacity - current_lr_capacity) if target_lr_capacity else 0

        if pad_amount_sr == 0 and pad_amount_lr == 0:
            # Still need to ensure LR arrays exist
            if target_lr_capacity and target_lr_capacity > 0:
                if not hasattr(neighbors, 'idx_i_lr') or neighbors.idx_i_lr is None:
                    # Create LR arrays filled with n_atoms (invalid index) for consistency
                    # with pad_leaf_with_path which also uses n_atoms as fill_value
                    neighbors = neighbors._replace(
                        idx_i_lr=jnp.full((target_lr_capacity,), self.n_atoms, dtype=jnp.int32),
                        idx_j_lr=jnp.full((target_lr_capacity,), self.n_atoms, dtype=jnp.int32)
                    )
            return neighbors

        def pad_leaf_with_path(path, arr):
            """Pad array based on its field name, not its size."""
            if not hasattr(arr, 'shape') or arr.ndim == 0:
                return arr
            
            # Extract field name from path (handles nested structures)
            field_name = None
            for key in reversed(path):
                if hasattr(key, 'key'):  # DictKey
                    field_name = key.key
                    break
                elif hasattr(key, 'name'):  # GetAttrKey (NamedTuple)
                    field_name = key.name
                    break
                elif isinstance(key, str):
                    field_name = key
                    break
            
            # CRITICAL: Skip per-atom fields - they should NOT be padded by edge capacity.
            # reference_positions has shape (n_atoms, 3), not (n_edges, 3).
            if field_name in _ATOM_FIELDS:
                return arr
            
            # Determine padding based on field name
            if field_name in _LR_FIELDS:
                # LR field: pad to LR capacity
                if target_lr_capacity is not None and arr.shape[0] < target_lr_capacity:
                    pad_width = [(0, target_lr_capacity - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                    # FIX: Pad with n_atoms (invalid index) instead of 0 (valid atom 0)
                    fill_value = self.n_atoms
                    return jnp.pad(arr, pad_width, mode='constant', constant_values=fill_value)
            else:
                # SR field: pad to SR capacity
                if arr.shape[0] < target_capacity:
                    pad_width = [(0, target_capacity - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                    # FIX: Pad with n_atoms (invalid index) instead of 0 (valid atom 0)
                    fill_value = self.n_atoms
                    return jnp.pad(arr, pad_width, mode='constant', constant_values=fill_value)
            return arr

        padded = jax.tree_util.tree_map_with_path(pad_leaf_with_path, neighbors)
        
        # Ensure LR arrays exist after padding
        if target_lr_capacity and target_lr_capacity > 0:
            if not hasattr(padded, 'idx_i_lr') or padded.idx_i_lr is None:
                # Use n_atoms (invalid index) for consistency with pad_leaf_with_path
                padded = padded._replace(
                    idx_i_lr=jnp.full((target_lr_capacity,), self.n_atoms, dtype=jnp.int32),
                    idx_j_lr=jnp.full((target_lr_capacity,), self.n_atoms, dtype=jnp.int32)
                )
        
        # CRITICAL FIX: Reset overflow flag after padding to prevent false overflow detection
        # GLP's update_fn checks if new neighbors fit in capacity, but uses internal capacity field.
        # After padding, we have more space, so overflow should be False.
        # NOTE: Use jnp.array(False) not Python False - Python bool lacks .ndim so gets stacked incorrectly
        if hasattr(padded, 'overflow'):
            padded = padded._replace(overflow=jnp.array(False))
        
        # Update capacity field if present (GLP stores this internally)
        if hasattr(padded, 'capacity'):
            padded = padded._replace(capacity=target_capacity)
        
        return padded

    def _stack_pytree_with_axes(self, pytree_list):
        """Stack list of pytrees, computing adaptive in_axes for vmap."""
        if not pytree_list:
            return None, None
        flat_pytrees = [jax.tree_util.tree_flatten(pt) for pt in pytree_list]
        tree_def = flat_pytrees[0][1]
        n_leaves = len(flat_pytrees[0][0])
        stacked_leaves = []
        in_axes_leaves = []
        for i in range(n_leaves):
            leaf_values = [flat[0][i] for flat in flat_pytrees]
            first_leaf = leaf_values[0]
            # Scalars get in_axes=None, arrays get in_axes=0
            if hasattr(first_leaf, 'ndim') and first_leaf.ndim == 0:
                stacked_leaves.append(first_leaf)
                in_axes_leaves.append(None)
            else:
                stacked_leaves.append(jnp.stack(leaf_values, axis=0))
                in_axes_leaves.append(0)
        stacked = jax.tree_util.tree_unflatten(tree_def, stacked_leaves)
        in_axes = jax.tree_util.tree_unflatten(tree_def, in_axes_leaves)
        return stacked, in_axes

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
        """Compute in_axes for neighbors pytree: arrays with batch dim get 0, others get None.
        
        After broadcast_to fix, all arrays including former 0-d scalars are batched:
        - Former 0-d scalar: now (n_batch,) → ndim=1 → in_axes=0
        - Former 1-d array: now (n_batch, N) → ndim=2 → in_axes=0
        Non-array leaves (None, python scalars) get in_axes=None.
        """
        flat, tree_def = jax.tree_util.tree_flatten(neighbors_batched)
        in_axes = [0 if (hasattr(leaf, 'ndim') and leaf.ndim > 0) else None for leaf in flat]
        return jax.tree_util.tree_unflatten(tree_def, in_axes)

    # =========================================================================
    # Compute Pipeline Steps
    # =========================================================================

    def _prepare_system_template(self, pos_ang_b, cell_ang_b):
        """Ensure system template and neighbor allocator are initialized.
        
        NOTE: i-PI uses lattice vectors as COLUMNS, but ASE/GLP/MLFF expect ROWS.
        We transpose the cell here to convert from i-PI convention to ASE convention.
        """
        if self._system_template is None:
            atoms = self.template_atoms.copy()
            atoms.set_positions(pos_ang_b[0], apply_constraint=False)
            # Transpose: i-PI (columns) -> ASE (rows)
            atoms.set_cell(cell_ang_b[0].T, scale_atoms=False)
            self._system_template = self._atoms_to_system(atoms, dtype=self.dtype)

        if not self._nl_initialized:
            # OPTIMIZED: Initialize GLP neighbor list directly, bypassing mlffCalculatorSparse.
            # This saves ~1-4s setup time and ~120-570MB GPU memory.
            from glp.neighborlist import quadratic_neighbor_list
            
            cell_init = self._system_template.cell
            self._neighbor_allocator, self._glp_update_fn = quadratic_neighbor_list(
                cell=cell_init, cutoff=self.cutoff, skin=self.skin,
                capacity_multiplier=self.capacity_multiplier, lr_cutoff=self.lr_cutoff
            )
            positions_init = jnp.array(pos_ang_b[0], dtype=self.dtype)
            self._neighbor_template = self._neighbor_allocator(positions_init)
            
            # Initialize LR capacity to prevent JIT retraces when distant molecules come together.
            # Conservative estimate: each atom could have ~n_atoms/2 LR neighbors on average.
            # This avoids shape changes from (0,) -> (N,) which trigger expensive recompilation.
            initial_lr_estimate = max(self.n_atoms * self.n_atoms // 4, 32)
            self._max_neighbor_lr_capacity_seen = max(
                self._max_neighbor_lr_capacity_seen, 
                initial_lr_estimate
            )
            
            # Also set initial SR capacity based on template
            if hasattr(self._neighbor_template, 'centers'):
                initial_sr_capacity = int(self._neighbor_template.centers.shape[0] * 1.25)
                self._max_neighbor_capacity_seen = max(self._max_neighbor_capacity_seen, initial_sr_capacity)
            
            # CRITICAL: Pad the initial template to estimated capacities.
            # This ensures the first JIT trace uses shapes that can accommodate future neighbors.
            self._neighbor_template = self._pad_neighbor_list(
                self._neighbor_template,
                self._max_neighbor_capacity_seen,
                self._max_neighbor_lr_capacity_seen
            )
            
            self._nl_initialized = True
            
            if self.verbose:
                print(f"[SO3LR] ✓ GLP neighbor list initialized (SR capacity: {self._max_neighbor_capacity_seen}, LR capacity: {self._max_neighbor_lr_capacity_seen})")

    def _update_neighbor_lists(self, positions_batched, cell_tensors_batched, n_batch, diagnostics=False):
        """Update neighbor lists every step using force_update=True.
        
        SIMPLIFIED ARCHITECTURE (Jan 2026):
        ===================================
        This method ALWAYS rebuilds neighbor lists using GLP's update_fn with force_update=True.
        
        Why we abandoned the skin-based optimization:
        - GLP's update_fn without force_update only recomputes distances for EXISTING pairs
        - It does NOT detect new pairs that come into interaction range as atoms move
        - This leads to ~1 meV/step energy errors and 20+ meV/ps drift in NVE
        - The official SO3LR jax_md example uses dr_threshold=0 (rebuild every step)
        
        Performance impact:
        - GLP update_fn with force_update=True: ~0.6 ms per step (JIT-compiled)
        - This is <5% overhead compared to model evaluation (~15 ms)
        - Much simpler and more robust than trying to detect topology changes
        
        The skin parameter is kept for GLP's internal buffer sizing but no longer
        controls rebuild frequency.
        """
        
        # Create stacked template if needed (first call or batch size change)
        if (self._stacked_template_neighbors is None or 
            self._n_batch_cached != n_batch or
            self._template_version != id(self._neighbor_template)):
            
            self._stacked_template_neighbors = jax.tree_util.tree_map(
                lambda x: _stack_leaf_broadcast(x, n_batch), self._neighbor_template
            )
            self._template_version = id(self._neighbor_template)
            self._neighbors_in_axes = self._compute_in_axes_for_neighbors(self._stacked_template_neighbors)
            
            if diagnostics:
                print(f"[SO3LR] Created stacked template neighbors for batch_size={n_batch}")
        
        # Create vmapped force-rebuild function if needed
        if not hasattr(self, '_vmapped_force_rebuild_fn') or self._vmapped_force_rebuild_fn is None:
            def force_rebuild_single(pos, nbrs, cell):
                """Force full neighbor recomputation with fixed capacity."""
                return self._glp_update_fn(pos, nbrs, new_cell=cell, force_update=True)
            
            self._vmapped_force_rebuild_fn = jax.jit(jax.vmap(
                force_rebuild_single,
                in_axes=(0, self._neighbors_in_axes, 0)
            ))
        
        # Always rebuild with force_update=True (captures new pairs every step)
        updated_neighbors_batched = self._vmapped_force_rebuild_fn(
            positions_batched,
            self._stacked_template_neighbors,
            cell_tensors_batched
        )
        
        # Check for overflow (capacity exceeded)
        overflow_flags = updated_neighbors_batched.overflow
        any_overflow = bool(jax.device_get(jnp.any(overflow_flags)))
        
        if any_overflow:
            # Overflow: need to grow capacity using allocate_fn (rare)
            print(f"[SO3LR] ⚠️ Overflow detected, growing capacity with allocate_fn")
            
            neighbors_list = self._unstack_pytree(updated_neighbors_batched, n_batch)
            overflow_np = np.asarray(overflow_flags)
            
            for i in range(n_batch):
                if overflow_np[i]:
                    neighbors_list[i] = self._neighbor_allocator(
                        positions_batched[i], new_cell=cell_tensors_batched[i]
                    )
            
            # Find new max capacities
            max_sr_cap = max(n.centers.shape[0] for n in neighbors_list)
            lr_caps = [n.idx_i_lr.shape[0] for n in neighbors_list
                       if hasattr(n, 'idx_i_lr') and n.idx_i_lr is not None]
            max_lr_cap = max(lr_caps) if lr_caps else 0
            
            # Grow capacity with buckets to minimize JIT recompilations
            SR_BUCKET = 500
            LR_BUCKET = 1000
            
            old_sr_cap = self._max_neighbor_capacity_seen
            old_lr_cap = self._max_neighbor_lr_capacity_seen
            
            self._max_neighbor_capacity_seen = max(self._max_neighbor_capacity_seen, 
                                                    _round_up_to_bucket(max_sr_cap, SR_BUCKET))
            self._max_neighbor_lr_capacity_seen = max(self._max_neighbor_lr_capacity_seen,
                                                       _round_up_to_bucket(max_lr_cap, LR_BUCKET))
            
            # Clear caches on shape change
            if self._max_neighbor_capacity_seen != old_sr_cap or self._max_neighbor_lr_capacity_seen != old_lr_cap:
                self._topology_jit = None
                self._model_jit = None
                self._cached_static_inputs = None
                self._vmapped_force_rebuild_fn = None
                self._stacked_template_neighbors = None
                print(f"[SO3LR] Capacity grown: SR={self._max_neighbor_capacity_seen}, LR={self._max_neighbor_lr_capacity_seen}")
            
            # Pad all neighbor lists to new capacity
            for i in range(len(neighbors_list)):
                neighbors_list[i] = self._pad_neighbor_list(
                    neighbors_list[i], self._max_neighbor_capacity_seen, self._max_neighbor_lr_capacity_seen
                )
            
            # Update template and restack
            self._neighbor_template = jax.tree_util.tree_map(lambda x: x, neighbors_list[0])
            self._template_version = id(self._neighbor_template)
            updated_neighbors_batched = jax.tree.map(lambda *args: jnp.stack(args), *neighbors_list)
            
            # Recreate stacked template for next iteration
            self._stacked_template_neighbors = jax.tree_util.tree_map(
                lambda x: _stack_leaf_broadcast(x, n_batch), self._neighbor_template
            )
        
        # Store batched neighbors
        self._batched_neighbors = updated_neighbors_batched
        self._n_batch_cached = n_batch
        
        # Always return needs_rebuild=True since we rebuild every step
        # This ensures _run_vmapped_calculation recomputes static inputs
        return updated_neighbors_batched, False  # False = needs_rebuild happened



    def _create_topology_jit(self):
        """Create JIT function to compute STATIC inputs (indices, masks).
        
        OPTIMIZATION: This runs ONLY on neighbor list rebuild.
        Output is cached and reused until next rebuild.
        
        Returns a function: (neighbors_batched) -> static_inputs_dict
        """
        dtype = self.dtype
        n_atoms = self.n_atoms
        
        def detect_valid_edges(idx_i, idx_j):
            """Detect valid edges (not padding, not self-loops) and return safe indices."""
            valid_mask = (
                (idx_i >= 0) & (idx_j >= 0) & 
                (idx_i < n_atoms) & (idx_j < n_atoms) & 
                (idx_i != idx_j)
            )
            safe_idx_i = jnp.where(valid_mask, idx_i, 0)
            safe_idx_j = jnp.where(valid_mask, idx_j, 0)
            return valid_mask, safe_idx_i, safe_idx_j
        
        def topology_single(neighbors):
            """Compute static inputs for a single bead."""
            idx_i = neighbors.centers
            idx_j = neighbors.others
            
            valid_sr, safe_idx_i, safe_idx_j = detect_valid_edges(idx_i, idx_j)
            edge_mask = valid_sr.astype(dtype)
            
            # Handle LR indices
            idx_i_lr = getattr(neighbors, 'idx_i_lr', None)
            idx_j_lr = getattr(neighbors, 'idx_j_lr', None)
            
            if idx_i_lr is None:
                idx_i_lr = jnp.zeros((0,), dtype=jnp.int32)
            if idx_j_lr is None:
                idx_j_lr = jnp.zeros((0,), dtype=jnp.int32)
            
            valid_lr, safe_idx_i_lr, safe_idx_j_lr = detect_valid_edges(idx_i_lr, idx_j_lr)
            edge_mask_lr = valid_lr.astype(dtype)
            
            return {
                'safe_idx_i': safe_idx_i,
                'safe_idx_j': safe_idx_j,
                'edge_mask': edge_mask,
                'valid_sr': valid_sr,
                'safe_idx_i_lr': safe_idx_i_lr,
                'safe_idx_j_lr': safe_idx_j_lr,
                'edge_mask_lr': edge_mask_lr,
                'valid_lr': valid_lr,
            }
        
        return jax.jit(jax.vmap(topology_single, in_axes=(self._neighbors_in_axes,)))
    
    def _create_model_jit(self):
        """Create JIT function for model execution using CACHED static inputs.
        
        OPTIMIZATION: Static inputs (indices, masks) are passed in, not recomputed.
        This runs EVERY step but with minimal overhead.
        
        Returns a function: (static_inputs, positions, cells) -> (E, F, S)
        """
        dtype = self.dtype
        so3lr_calc = self.so3lr_calc
        n_atoms = self.n_atoms
        total_charge = self.total_charge  # int
        num_unpaired_electrons = self.num_unpaired_electrons  # int
        theory_mask_const = self._theory_mask_const  # float32 one-hot (1, num_theory_levels)
        calculate_stress = self.calc_stress
        atomic_numbers_const = jnp.array(self.atomic_numbers, dtype=jnp.int32)
        
        # INVALID_OFFSET: Large offset to push padded SR edges far away.
        # 
        # HOW IT WORKS (SR only):
        # - Invalid SR edges get offset [10000, 10000, 10000]
        # - This produces r_ij ≈ 300,000 Å after PBC wrapping (10000 * cell_size)
        # - cutoff_fn(300,000) → 0.0 (cutoff is ~4.5 Å)
        # - edge_mask (patched mlff) multiplies cutoff → double-zeroed
        #
        # NOTE: This does NOT affect LR edges!
        # - mlff IGNORES cell_offset_lr due to shape mismatch (n_edges_sr vs n_edges_lr)
        # - LR invalid edges are handled via: edge_mask_lr applied to final energy
        # 
        # REQUIRES: mlff branch feat/full-sparse-padding-support
        INVALID_OFFSET = jnp.array([10000, 10000, 10000], dtype=jnp.int32)
        # LR edges use zero offset for invalid entries (not INVALID_OFFSET)
        # because large offsets × cell produce distances that overflow R^10 in dispersion backward pass
        ZERO_OFFSET = jnp.array([0, 0, 0], dtype=jnp.int32)
        
        def compute_offsets(positions, safe_idx_i, safe_idx_j, inv_cell, valid_mask, invalid_offset):
            """Compute integer cell offsets for SR or LR edges.
            
            Uses pre-computed inv(cell) to avoid redundant inversion.
            disp_frac = (r_j - r_i) @ inv_cell
            offset = -round(disp_frac)
            
            Args:
                inv_cell: Pre-computed jnp.linalg.inv(cell)
                invalid_offset: INVALID_OFFSET for SR, ZERO_OFFSET for LR
            """
            r_i = positions[safe_idx_i]
            r_j = positions[safe_idx_j]
            disp_frac = jnp.dot(r_j - r_i, inv_cell)
            offset = -jnp.round(disp_frac).astype(jnp.int32)
            return jnp.where(valid_mask[:, None], offset, invalid_offset)
        
        def model_single(static_inputs, positions, cell):
            """Execute model for a single bead using cached static inputs.
            
            Note: atomic_numbers captured via closure (compile-time constant).
            """
            # Unpack static inputs (NO recomputation!)
            safe_idx_i = static_inputs['safe_idx_i']
            safe_idx_j = static_inputs['safe_idx_j']
            edge_mask = static_inputs['edge_mask']
            valid_sr = static_inputs['valid_sr']
            safe_idx_i_lr = static_inputs['safe_idx_i_lr']
            safe_idx_j_lr = static_inputs['safe_idx_j_lr']
            edge_mask_lr = static_inputs['edge_mask_lr']
            valid_lr = static_inputs['valid_lr']
            
            # Deduce n_edges from array shape (static integer inside vmap)
            n_edges = safe_idx_i.shape[0]
            n_edges_lr = safe_idx_i_lr.shape[0]  # LR edge count for cell_lr broadcast
            
            # Dynamic: Compute cell offsets for PBC (used by model's internal displacement calculation)
            # OPTIMIZATION: Invert cell ONCE (not twice for SR + LR)
            inv_cell = jnp.linalg.inv(cell)
            final_offset = compute_offsets(positions, safe_idx_i, safe_idx_j, inv_cell, valid_sr, INVALID_OFFSET)
            final_offset_lr = compute_offsets(positions, safe_idx_i_lr, safe_idx_j_lr, inv_cell, valid_lr, ZERO_OFFSET)
            
            # NOTE: Model uses input_convention='positions' (default from load_model_from_workdir)
            # so it computes displacements internally from positions + cell_offset.
            # We don't need to pass 'displacements' - only 'positions' and 'cell_offset'.

            inputs = {
                'positions': positions,
                'atomic_numbers': atomic_numbers_const,  # Captured constant
                'cell': jnp.broadcast_to(cell[None, :, :], (n_edges, 3, 3)),
                'cell_per_atom': jnp.broadcast_to(cell[None, :, :], (n_atoms, 3, 3)),
                'idx_i': safe_idx_i,
                'idx_j': safe_idx_j,
                'cell_offset': final_offset,
                'node_mask': jnp.ones((n_atoms,), dtype=dtype),
                'edge_mask': edge_mask,
                # FIX: Use int16 for charge/spin (matches mlff dataloader), one-hot float32 for theory_mask
                'total_charge': jnp.array([total_charge], dtype=jnp.int16),
                'num_unpaired_electrons': jnp.array([num_unpaired_electrons], dtype=jnp.int16),
                'theory_mask': theory_mask_const,  # Pre-computed one-hot float32
                'batch_segments': jnp.zeros(n_atoms, dtype=jnp.int32),
                'graph_mask': jnp.array([True]),
                'idx_i_lr': safe_idx_i_lr,
                'idx_j_lr': safe_idx_j_lr,
                'cell_offset_lr': final_offset_lr,
                'cell_lr': jnp.broadcast_to(cell[None, :, :], (n_edges_lr, 3, 3)),  # LR cell for PBC
                'edge_mask_lr': edge_mask_lr,
            }

            
            # Run model
            output = so3lr_calc(inputs)
            
            # Unit conversion on GPU
            energy_hartree = output['energy'] * EV_TO_HARTREE
            forces_hartree_bohr = output['forces'] * (EV_TO_HARTREE * BOHR_TO_ANG)
            
            # Stress: eV/Ang^3 -> Hartree/Bohr^3 + Voigt
            # NOTE: This block is prepared for future NPT support. Currently, So3lr
            # never outputs 'stress', so we always return zeros. See calc_stress comment above.
            if calculate_stress and 'stress' in output:
                stress_raw = output['stress'] # (1, 6) or (1, 3, 3)
                # Squeeze batch dim if present (usually (1, ...))
                stress_squeezed = jnp.squeeze(stress_raw, axis=0)
                
                if stress_squeezed.shape == (6,):
                    s = stress_squeezed
                    stress_full = jnp.array([
                        [s[0], s[5], s[4]],
                        [s[5], s[1], s[3]],
                        [s[4], s[3], s[2]]
                    ])
                elif stress_squeezed.shape == (3, 3):
                    stress_full = stress_squeezed
                else:
                    stress_full = jnp.zeros((3, 3), dtype=dtype)
                
                stress_conv_factor = EV_TO_HARTREE * (BOHR_TO_ANG ** 3)
                stress_hartree_bohr = stress_full * stress_conv_factor
            else:
                stress_hartree_bohr = jnp.zeros((3, 3), dtype=dtype)
            
            return energy_hartree, forces_hartree_bohr, stress_hartree_bohr
        
        # in_axes: static_inputs dict batched, positions batched, cells batched
        # Note: atomic_numbers captured via closure, not passed as arg
        static_in_axes = {k: 0 for k in ['safe_idx_i', 'safe_idx_j', 'edge_mask', 'valid_sr',
                                          'safe_idx_i_lr', 'safe_idx_j_lr', 'edge_mask_lr', 'valid_lr']}
        return jax.jit(jax.vmap(model_single, in_axes=(static_in_axes, 0, 0)))

    def _run_vmapped_calculation(self, batched_system, batched_neighbors, n_batch, needs_rebuild=False):
        """Execute vmapped calculation with SPLIT JIT architecture.
        
        OPTIMIZED: 
        - Topology JIT (static inputs) runs ONLY on rebuild.
        - Model JIT (dynamic inputs) runs EVERY step using cached static inputs.
        """
        
        # Lazy initialization of JIT functions
        if self._topology_jit is None:
            self._topology_jit = self._create_topology_jit()
        if self._model_jit is None:
            self._model_jit = self._create_model_jit()
        
        # =====================================================================
        # STATIC INPUT CACHING (Topology JIT runs only on rebuild)
        # =====================================================================
        if needs_rebuild or self._cached_static_inputs is None:
            # Compute static inputs (indices, masks) - runs topology_jit
            self._cached_static_inputs = self._topology_jit(batched_neighbors)
        
        # =====================================================================
        # MODEL EXECUTION (uses cached static inputs)
        # =====================================================================
        energies_batched, forces_batched, stresses_batched = self._model_jit(
            self._cached_static_inputs,
            batched_system.R,           # Positions (dynamic)
            batched_system.cell         # Cells (dynamic)
            # atomic_numbers captured via closure in _create_model_jit
        )
        
        # Transfer results to CPU
        energies_k, forces_k, stresses_k = jax.device_get(
            (energies_batched, forces_batched, stresses_batched)
        )
        
        # Package results
        result_list = [
            (float(energies_k[i]), forces_k[i].ravel(), stresses_k[i], self._empty_json)
            for i in range(n_batch)
        ]

        return result_list

    # =========================================================================
    # Main Interface
    # =========================================================================

    def compute_batch(self, cell_list, pos_list):
        """Main batch compute function using vmap parallelization."""
        start = time.time()
        n_batch = len(pos_list)
        self.eval_count += 1
        diagnostics = (self.eval_count <= 3) or (self.eval_count % 50 == 0)

        if diagnostics:
            print(f"[SO3LR] Batch #{self.eval_count}: {n_batch} structures")
        
        # Pre-allocate zero stresses if needed (avoids allocation in hot path)
        if self._zero_stresses is None or self._zero_stresses.shape[0] != n_batch:
            self._zero_stresses = np.zeros((n_batch, 3, 3), dtype=self.dtype)

        # 1. Unit Conversion (HIGH-PRECISION)
        # Perform Bohr → Angstrom conversion in float64 to preserve maximum precision.
        # Casting to float32 BEFORE multiplication introduces quantization error in:
        #   1. Input positions (Bohr) quantized from i-PI's float64
        #   2. The conversion factor itself
        # By converting in float64 first, we preserve ~15 digits of precision during
        # the arithmetic, only truncating to float32 when creating JAX arrays.
        cell_ang_b = (np.asarray(cell_list, dtype=np.float64) * BOHR_TO_ANG).astype(self.dtype)
        pos_ang_b = (np.asarray(pos_list, dtype=np.float64) * BOHR_TO_ANG).astype(self.dtype)

        # 2. Prepare JAX arrays
        # NOTE: i-PI uses lattice vectors as COLUMNS, but ASE/GLP/MLFF expect ROWS.
        # Transpose (0, 2, 1) converts each cell from i-PI (columns) to ASE (rows).
        positions_batched = jnp.array(pos_ang_b, dtype=self.dtype)
        cell_tensors_batched = jnp.array(np.transpose(cell_ang_b, (0, 2, 1)), dtype=self.dtype)

        # 3. Initialize system/NL machinery
        self._prepare_system_template(pos_ang_b, cell_ang_b)

        # 4. Create or REUSE batched system object (OPTIMIZED)
        # Only tile on first call or batch size change - reuse cached template otherwise
        if self._cached_batched_system is not None and n_batch == self._n_batch_cached:
            # FAST PATH: Reuse cached static fields (Z, masses, etc.)
            batched_system = self._cached_batched_system
        else:
            # SLOW PATH: Initial batching (runs only once per batch size)
            # CRITICAL FIX: Use broadcast_to to add a NEW batch dimension at axis 0.
            # Previous tile((n,1,1)) was WRONG - it concatenated along axis 0:
            #   tile((162,3), (32,1,1)) → (32*162, 3) = (5184, 3)  ← WRONG
            # broadcast_to correctly creates:
            #   broadcast_to((162,3), (32,162,3)) → (32, 162, 3)  ← CORRECT
            batched_system = jax.tree_util.tree_map(
                lambda x: _stack_leaf_broadcast(x, n_batch), self._system_template
            )
            if diagnostics:
                print(f"[SO3LR] Created new batched system template for batch_size={n_batch}")
        
        # Update ONLY dynamic fields (positions and cell) - static fields stay cached
        batched_system = batched_system._replace(
             R=positions_batched,
             cell=cell_tensors_batched
        )
        
        if self._system_in_axes is None:
            # All fields are now batched along axis 0
            self._system_in_axes = jax.tree_util.tree_map(lambda x: 0, batched_system)

        # 5. Update neighbor lists
        neighbors_list, can_reuse_cache = self._update_neighbor_lists(
            positions_batched, cell_tensors_batched, n_batch, diagnostics
        )

        # 6. Use batched neighbors directly (already stored in _batched_neighbors by _update_neighbor_lists)
        # OPTIMIZED: Skip _prepare_batched_neighbors overhead
        batched_neighbors = self._batched_neighbors
        
        # Determine if static inputs need refresh (rebuild occurred)
        needs_rebuild = not can_reuse_cache

        if diagnostics and can_reuse_cache:
            print("[SO3LR] ✓ Fast Path: Reusing cached static inputs")

        # 7. Execute with split JIT (topology cached, model runs every step)
        results = self._run_vmapped_calculation(batched_system, batched_neighbors, n_batch, needs_rebuild=needs_rebuild)

        if diagnostics:
            t_total = time.time() - start
            print(f"[SO3LR] Total: {t_total:.3f}s ({t_total/n_batch:.4f}s/struct)")

        # Update cache
        self._cached_batched_system = batched_system
        self._n_batch_cached = n_batch

        return results

    def compute_structure(self, cell, pos):
        """Single structure evaluation (wraps batch)."""
        return self.compute_batch([cell], [pos])[0]

    def compute(self, cell, pos):
        """Main dispatch - handles both single and batch."""
        pos_arr = np.asarray(pos)
        if pos_arr.ndim == 3:
            return self.compute_batch(cell, pos)
        return self.compute_structure(cell, pos)

    def __call__(self, cell, pos):
        """Function interface required by i-pi."""
        return self.compute(cell, pos)
