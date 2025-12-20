"""Refactored SO3LR driver using mlff's vmap-based batching with Direct Optimization.

ARCHITECTURE OVERVIEW:
=====================
This driver integrates the SO3LR machine learning force field with i-PI's Path Integral
Molecular Dynamics (PIMD) engine. It uses jax.vmap to parallelize over PIMD beads.

Performance Optimization:
- Bypasses mlffCalculatorSparse.calculate_fn overhead.
- Calls So3lr model directly with on-device graph preparation.
- Implements "Robust Fast Path" offset calculation on GPU (same as jraph_main).

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

# Deferred imports
jax = None
jnp = None

from ipi.utils import units

# Constants
BOHR_TO_ANG = units.unit_to_user("length", "angstrom", 1.0)
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = units.unit_to_user("energy", "electronvolt", 1.0)
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

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
        self.calculator = None
        self.so3lr_calc = None  # Direct model
        self.atoms_to_system = None
        self.System = None

        # Template and data
        self.template_atoms = None
        self.atomic_numbers = None
        self.n_atoms = 0
        self._system_template = None

        # Neighbor list machinery
        self._spatial_partitioning = None
        self._neighbor_template = None
        self._neighbor_allocator = None
        self._update_neighbors_fn = None
        self._glp_allocate_fn = None
        self._glp_update_fn = None
        self._use_glp_direct = True

        # Vmapped functions
        self._vmapped_nl_update_fn = None
        self._neighbors_in_axes = None
        self._system_in_axes = None
        
        # GPU Topology Updater (optimization)
        self._unified_model_jit = None  # Single unified dispatch (topology + model)

        # State / Cache
        self._batched_neighbors = None
        self._n_batch_cached = 0
        self._stacked_template_neighbors = None  # Cached stacked template for NL update
        self._template_version = None  # Track when template changes (via id())
        
        # Fast Path cache
        self._cached_batched_system = None  # Cached batched System (static fields reused, dynamic fields overwritten)
        self._consecutive_no_rebuild = 0
        
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
        """Initialize the SO3LR calculator and parameters."""
        global jax, jnp
        import jax
        import jax.numpy as jnp

        from mlff.md import mlffCalculatorSparse
        from glp import System, atoms_to_system
        from so3lr import So3lr  # Import direct model class
        import so3lr as so3lr_pkg

        if self.verbose:
            print(f"[SO3LR] JAX devices: {jax.devices()}")

        self.atoms_to_system = atoms_to_system
        self.System = System

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
        self.calc_stress = self.kwargs.get('calculate_stress', False)
        self.damping = float(self.kwargs.get('dispersion_energy_cutoff_lr_damping', 2.0))
        self.total_charge = float(self.kwargs.get('total_charge', 0.0))
        self.num_unpaired_electrons = float(self.kwargs.get('num_unpaired_electrons', 0.0))

        # Neighbor list parameters
        self.skin = float(self.kwargs.get('skin', 1.0))
        self.capacity_multiplier = float(self.kwargs.get('capacity_multiplier', 1.25))
        self.buffer_size_multiplier = float(self.kwargs.get('buffer_size_multiplier', 1.25))

        # Model path
        model_path = self.kwargs.get('model_path')
        if model_path:
            params_dir = pathlib.Path(model_path)
        else:
            params_dir = pathlib.Path(so3lr_pkg.__file__).parent / 'params'
            if not params_dir.exists():
                raise ImportError(f"Could not find params at {params_dir}. Provide 'model_path'.")

        # 1. Neighbor List Calculator (MLFF) - Used ONLY for NL updates
        self.calculator = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=params_dir,
            lr_cutoff=self.lr_cutoff,
            from_file=True,
            calculate_stress=self.calc_stress,
            dtype=self.dtype,
            skin=self.skin,
            capacity_multiplier=self.capacity_multiplier,
            buffer_size_multiplier=self.buffer_size_multiplier,
        )
        
        # 2. SO3LR Calculator - Used for actual energy/force calculation
        self.so3lr_calc = So3lr(
            calculate_forces=True,
            lr_cutoff=self.lr_cutoff
        )

        # num_theory_levels: Required for theory_mask shape
        # Default to 16 (standard SO3LR model), override via kwargs if different
        self.num_theory_levels = int(self.kwargs.get('num_theory_levels', 16))
        if self.verbose:
            print(f"[SO3LR] Using num_theory_levels: {self.num_theory_levels}")

        if self.verbose:
            print("[SO3LR] ✓ Initialized mlffCalculatorSparse (for NL) and So3lr (for Compute)")

    # =========================================================================
    # Helpers
    # =========================================================================

    # Fields that use LR capacity (explicit list to avoid size-based ambiguity)
    _LR_FIELDS = frozenset({'idx_i_lr', 'idx_j_lr', 'cell_offset_lr', 'distance_lr'})
    
    # Per-ATOM fields that should NOT be padded by edge capacity.
    # These arrays have shape (n_atoms, ...) not (n_edges, ...).
    _ATOM_FIELDS = frozenset({'reference_positions', 'positions'})

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
                    # Create empty LR arrays with target capacity
                    neighbors = neighbors._replace(
                        idx_i_lr=jnp.zeros((target_lr_capacity,), dtype=jnp.int32),
                        idx_j_lr=jnp.zeros((target_lr_capacity,), dtype=jnp.int32)
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
            
            if self.verbose:
                 print(f"DEBUG: Processing field '{field_name}', shape={arr.shape}")

            # CRITICAL: Skip per-atom fields - they should NOT be padded by edge capacity.
            # reference_positions has shape (n_atoms, 3), not (n_edges, 3).
            if field_name in self._ATOM_FIELDS:
                if self.verbose:
                    print(f"DEBUG: SKIPPING padding for atom field '{field_name}'")
                return arr
            
            # Determine padding based on field name
            if field_name in self._LR_FIELDS:
                # LR field: pad to LR capacity
                if target_lr_capacity is not None and arr.shape[0] < target_lr_capacity:
                    pad_width = [(0, target_lr_capacity - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                    return jnp.pad(arr, pad_width, mode='constant', constant_values=0)
            else:
                # SR field: pad to SR capacity
                if arr.shape[0] < target_capacity:
                    pad_width = [(0, target_capacity - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                    return jnp.pad(arr, pad_width, mode='constant', constant_values=0)
            return arr

        padded = jax.tree_util.tree_map_with_path(pad_leaf_with_path, neighbors)
        
        # Ensure LR arrays exist after padding
        if target_lr_capacity and target_lr_capacity > 0:
            if not hasattr(padded, 'idx_i_lr') or padded.idx_i_lr is None:
                padded = padded._replace(
                    idx_i_lr=jnp.zeros((target_lr_capacity,), dtype=jnp.int32),
                    idx_j_lr=jnp.zeros((target_lr_capacity,), dtype=jnp.int32)
                )
        
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

    def _voigt_to_full_stress(self, stress_voigt):
        """Convert stress from Voigt notation (6,) to full 3x3 tensor."""
        if stress_voigt.shape == (3, 3):
            return stress_voigt
        elif stress_voigt.shape == (6,):
            return np.array([
                [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                [stress_voigt[4], stress_voigt[3], stress_voigt[2]]
            ])
        return np.zeros((3, 3))

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
            self._system_template = self.atoms_to_system(atoms, dtype=self.dtype)

        if self._spatial_partitioning is None:
            # Initialize via calculator.calculate() to set up internal state
            atoms = self.template_atoms.copy()
            atoms.set_positions(pos_ang_b[0], apply_constraint=False)
            # Transpose: i-PI (columns) -> ASE (rows)
            atoms.set_cell(cell_ang_b[0].T, scale_atoms=False)
            self.calculator.calculate(atoms=atoms, properties=['energy', 'forces'])

            self._spatial_partitioning = self.calculator.spatial_partitioning
            self._neighbor_template = jax.tree_util.tree_map(lambda x: x, self.calculator.neighbors)

            if self._use_glp_direct:
                from glp.neighborlist import quadratic_neighbor_list
                cell_init = self._system_template.cell
                self._glp_allocate_fn, self._glp_update_fn = quadratic_neighbor_list(
                    cell=cell_init, cutoff=self.cutoff, skin=self.skin,
                    capacity_multiplier=self.capacity_multiplier, lr_cutoff=self.lr_cutoff
                )
                positions_init = jnp.array(pos_ang_b[0], dtype=self.dtype)
                self._neighbor_template = self._glp_allocate_fn(positions_init)
                self._update_neighbors_fn = self._glp_update_fn
                self._neighbor_allocator = self._glp_allocate_fn
            else:
                self._neighbor_allocator = self._spatial_partitioning.allocate_fn
                self._update_neighbors_fn = self._spatial_partitioning.update_fn
            
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
            
            if self.verbose:
                print(f"[SO3LR] Initial SR capacity: {self._max_neighbor_capacity_seen}, LR capacity: {self._max_neighbor_lr_capacity_seen}")

    def _update_neighbor_lists(self, positions_batched, cell_tensors_batched, n_batch, diagnostics=False):
        """Update neighbor lists on GPU. Syncs to CPU only on overflow or skin violation.
        
        IMPORTANT: Skin violations MUST trigger a full neighbor list rebuild because:
        - GLP's update_fn only recomputes distances for EXISTING pairs
        - New neighbor pairs that entered the cutoff won't be captured
        - This would lead to missing neighbors → incorrect forces
        
        The optimization keeps data on GPU when no rebuild is needed.
        """
        
        # OPTIMIZED: Cache the stacked template to avoid Python list creation every step
        # We must recreate from template (not reuse vmapped output) because:
        # 1. _neighbor_template is updated after rebuilds with new reference positions
        # 2. GLP's update_fn uses internal cond() that requires fresh structure
        #
        # We invalidate cache when: template changes (after rebuild) OR batch size changes
        if self._batched_neighbors is not None and self._n_batch_cached == n_batch:
             # CRITICAL FIX: Use existing batched state which has correct reference_positions
             # per bead. Using the template would reset all beads to Bead 0's reference,
             # causing immediate skin violations for other beads in PIMD.
             prev_neighbors_batched = self._batched_neighbors
        else:
             # Only create/use stacked template on first call or batch size change
             if (self._stacked_template_neighbors is None or 
                 self._n_batch_cached != n_batch or
                 self._template_version != id(self._neighbor_template)):
                 
                 # Create stacked version from template
                 # CRITICAL FIX: Use broadcast_to to ensure ALL arrays (including 0-d scalars)
                 # are properly batched. This prevents in_axes inconsistency when:
                 # 1. First call: template has 0-d scalars (e.g., overflow) → in_axes=None
                 # 2. Subsequent calls: _batched_neighbors may have (n_batch,) → in_axes mismatch
                 # 
                 # broadcast_to(x, (n_batch,) + x.shape) correctly handles:
                 # - 0-d scalar: shape=() → (n_batch,)  
                 # - 1-d array: shape=(N,) → (n_batch, N)
                 # - 2-d array: shape=(N, M) → (n_batch, N, M)
                 def stack_leaf(x):
                     if hasattr(x, 'ndim'):
                         return jnp.broadcast_to(x, (n_batch,) + x.shape)
                     return x  # Non-array leaf (e.g., None, python scalar)
                 
                 self._stacked_template_neighbors = jax.tree_util.tree_map(
                     stack_leaf,
                     self._neighbor_template
                 )
                 self._template_version = id(self._neighbor_template)
                 if diagnostics:
                     print(f"[SO3LR] Created stacked template neighbors for batch_size={n_batch}")
             
             prev_neighbors_batched = self._stacked_template_neighbors

        # Create vmapped NL update if needed
        if self._vmapped_nl_update_fn is None:
            self._neighbors_in_axes = self._compute_in_axes_for_neighbors(prev_neighbors_batched)
            skin_threshold = self.skin / 2.0

            def nl_update_single(pos, nbrs, cell):
                # Check skin violation: atoms moved too far from reference positions
                ref_pos = nbrs.reference_positions
                max_displacement = jnp.max(jnp.linalg.norm(pos - ref_pos, axis=-1))
                skin_violation = max_displacement > skin_threshold
                
                # FIX: Call GLP's update_fn to ensure internal state (like reference_positions) 
                # is properly managed and valid updates are performed.
                # Previous versions avoided this due to concerns about vmap(cond), but 
                # JAX handles scalar predicates inside vmap correctly.
                # Skipping this caused "Ghost Updates" where the neighbor list logic was 
                # completely bypassed until a manual skin violation check forced a rebuild.
                updated_nbrs = self._update_neighbors_fn(pos, nbrs, new_cell=cell)

                return updated_nbrs, skin_violation

            self._vmapped_nl_update_fn = jax.jit(jax.vmap(
                nl_update_single,
                in_axes=(0, self._neighbors_in_axes, 0)
            ))

        # Run update (stays on GPU)
        updated_neighbors_batched, skin_violations = self._vmapped_nl_update_fn(
            positions_batched, prev_neighbors_batched, cell_tensors_batched
        )

        # Check overflow and skin violations (single sync point for both)
        overflow_flags = getattr(updated_neighbors_batched, "overflow", None)
        
        # Sync both at once to minimize CPU-GPU transfers
        if overflow_flags is not None:
            any_overflow, any_skin_violation = jax.device_get(
                (jnp.any(overflow_flags), jnp.any(skin_violations))
            )
            any_overflow = bool(any_overflow)
            any_skin_violation = bool(any_skin_violation)
        else:
            any_overflow = False
            any_skin_violation = bool(jax.device_get(jnp.any(skin_violations)))

        needs_rebuild = any_overflow or any_skin_violation

        # Handle overflow or skin violation: need to rebuild neighbor list (full search)
        if needs_rebuild:
            if diagnostics:
                reason = "overflow" if any_overflow else "skin violation"
                print(f"[SO3LR] ⚠️ {reason.capitalize()} detected, rebuilding neighbor lists")
            
            # Must go to CPU for rebuild (full neighbor search)
            neighbors_list = self._unstack_pytree(updated_neighbors_batched, n_batch)
            overflow_np = np.asarray(overflow_flags) if overflow_flags is not None else np.zeros(n_batch, dtype=bool)
            skin_np = np.asarray(skin_violations)
            
            for i, neighbors in enumerate(neighbors_list):
                if overflow_np[i] or skin_np[i]:
                    # Rebuild this bead's neighbor list (full pairwise search)
                    # FIX: positions_batched[i] is already a JAX device array.
                    # Avoid np.asarray() roundtrip which causes GPU→CPU sync + CPU→GPU transfer.
                    pos_jax = positions_batched[i]
                    # Pass new_cell directly to allocate (avoids redundant update call)
                    neighbors_list[i] = self._neighbor_allocator(
                        pos_jax, new_cell=cell_tensors_batched[i]
                    )
            
            # Find max capacities across ALL beads (rebuilt ones may differ from others)
            max_sr_cap = max(n.centers.shape[0] for n in neighbors_list)
            max_lr_cap = 0
            # FIXED: Check ALL beads, not just bead 0. Some beads may have idx_i_lr=None.
            lr_caps = [
                n.idx_i_lr.shape[0] for n in neighbors_list
                if hasattr(n, 'idx_i_lr') and n.idx_i_lr is not None
            ]
            if lr_caps:
                max_lr_cap = max(lr_caps)
            
            # Update capacity tracking with buffer
            self._max_neighbor_capacity_seen = max(self._max_neighbor_capacity_seen, int(max_sr_cap * 1.25))
            self._max_neighbor_lr_capacity_seen = max(self._max_neighbor_lr_capacity_seen, int(max_lr_cap * 1.25))
            
            # Pad ALL neighbor lists to the same capacity before stacking
            for i in range(len(neighbors_list)):
                neighbors_list[i] = self._pad_neighbor_list(
                    neighbors_list[i], self._max_neighbor_capacity_seen, self._max_neighbor_lr_capacity_seen
                )
            
            # Update template
            self._neighbor_template = jax.tree_util.tree_map(lambda x: x, neighbors_list[0])
            
            # Restack after rebuild
            updated_neighbors_batched = jax.tree.map(lambda *args: jnp.stack(args), *neighbors_list)
            self._consecutive_no_rebuild = 0
        else:
            self._consecutive_no_rebuild += 1
        
        # Store batched neighbors (GPU-resident)
        self._batched_neighbors = updated_neighbors_batched
        self._n_batch_cached = n_batch
        
        return updated_neighbors_batched, not needs_rebuild



    def _create_unified_model_jit(self):
        """Create UNIFIED JIT function: topology prep + cell inversion + model in ONE dispatch.
        
        OPTIMIZATION: Consolidates 3 separate GPU dispatches into 1:
        - Topology preparation (mask detection, index clamping)
        - Batched cell inversion 
        - Model execution
        
        This eliminates 2 Python dispatch round-trips, significantly improving GPU utilization.
        """
        dtype = self.dtype
        so3lr_calc = self.so3lr_calc
        n_atoms = self.n_atoms
        total_charge = self.total_charge
        num_unpaired_electrons = self.num_unpaired_electrons
        num_theory_levels = self.num_theory_levels
        
        # Constant for invalid/masked offsets (large value to ensure exclusion)
        INVALID_OFFSET = jnp.array([10000, 10000, 10000], dtype=jnp.int32)
        
        # =====================================================================
        # HELPER: Detect valid edges and compute safe indices
        # =====================================================================
        def detect_valid_edges(idx_i, idx_j):
            """Detect valid edges (not padding, not self-loops) and return safe indices.
            
            Args:
                idx_i: Center atom indices (may contain invalid values from padding)
                idx_j: Neighbor atom indices
                
            Returns:
                valid_mask: Boolean mask of valid edges
                safe_idx_i: Indices clamped to valid range (invalid -> 0)
                safe_idx_j: Indices clamped to valid range (invalid -> 0)
            """
            valid_mask = (
                (idx_i >= 0) & (idx_j >= 0) & 
                (idx_i < n_atoms) & (idx_j < n_atoms) & 
                (idx_i != idx_j)
            )
            # Redirect invalid indices to 0 (creates self-loops for padding).
            # This is safe: compute_offsets applies INVALID_OFFSET for these edges,
            # resulting in d_ij >> cutoff, so they contribute zero energy.
            safe_idx_i = jnp.where(valid_mask, idx_i, 0)
            safe_idx_j = jnp.where(valid_mask, idx_j, 0)
            return valid_mask, safe_idx_i, safe_idx_j
        
        # =====================================================================
        # HELPER: Compute periodic cell offsets for minimum image convention
        # =====================================================================
        def compute_offsets(positions, safe_idx_i, safe_idx_j, inv_cell, valid_mask):
            """Compute integer cell offsets for periodic boundary conditions.
            
            Args:
                positions: Atomic positions (n_atoms, 3)
                safe_idx_i: Safe center indices
                safe_idx_j: Safe neighbor indices
                inv_cell: Inverse of cell matrix
                valid_mask: Boolean mask of valid edges
                
            Returns:
                offsets: Integer cell offsets (n_edges, 3), masked invalid -> INVALID_OFFSET
            """
            r_i = positions[safe_idx_i]
            r_j = positions[safe_idx_j]
            disp_raw = r_j - r_i
            # Cell is row-major (ASE convention): rows are lattice vectors a, b, c.
            # For r = s @ H, fractional coords are: s = r @ H^-1 (no transpose needed)
            disp_frac = jnp.dot(disp_raw, inv_cell)
            offset = -jnp.round(disp_frac).astype(jnp.int32)
            return jnp.where(valid_mask[:, None], offset, INVALID_OFFSET)
        
        # =====================================================================
        # MAIN: Unified model function for single bead
        # =====================================================================
        def unified_model_single(system, neighbors):
            """Complete pipeline for one bead: topology -> offsets -> model."""
            
            # --- Step 1: Process SR edges ---
            idx_i = neighbors.centers
            idx_j = neighbors.others
            n_edges = idx_i.shape[0]
            
            valid_sr, safe_idx_i, safe_idx_j = detect_valid_edges(idx_i, idx_j)
            edge_mask = valid_sr.astype(dtype)
            
            # --- Step 2: Process LR edges SAFELY ---
            # FIXED: Handle missing or None LR indices by normalizing to empty arrays.
            # This prevents AttributeError when LR interactions are not configured.
            idx_i_lr = getattr(neighbors, 'idx_i_lr', None)
            idx_j_lr = getattr(neighbors, 'idx_j_lr', None)
            
            # Normalize None → empty arrays with shape (0,)
            if idx_i_lr is None:
                idx_i_lr = jnp.zeros((0,), dtype=jnp.int32)
            if idx_j_lr is None:
                idx_j_lr = jnp.zeros((0,), dtype=jnp.int32)
                
            n_edges_lr = idx_i_lr.shape[0]
            
            valid_lr, safe_idx_i_lr, safe_idx_j_lr = detect_valid_edges(idx_i_lr, idx_j_lr)
            
            # --- Step 3: Cell inversion (XLA optimizes within vmap) ---
            inv_cell = jnp.linalg.inv(system.cell)
            
            # --- Step 4: Compute offsets ---
            final_offset = compute_offsets(system.R, safe_idx_i, safe_idx_j, inv_cell, valid_sr)
            
            # --- Step 5: Build model inputs ---
            inputs = {
                'positions': system.R,
                'atomic_numbers': system.Z,
                'cell': jnp.broadcast_to(system.cell[None, :, :], (n_edges, 3, 3)),
                'cell_per_atom': jnp.broadcast_to(system.cell[None, :, :], (n_atoms, 3, 3)),
                'idx_i': safe_idx_i,
                'idx_j': safe_idx_j,
                'cell_offset': final_offset,
                'node_mask': jnp.ones((n_atoms,), dtype=dtype),
                'edge_mask': edge_mask,
                'total_charge': jnp.array([total_charge], dtype=dtype),
                'num_unpaired_electrons': jnp.array([num_unpaired_electrons], dtype=dtype),
                'theory_mask': jnp.ones((1, num_theory_levels), dtype=jnp.int32),
                'batch_segments': jnp.zeros(n_atoms, dtype=jnp.int32),
                'graph_mask': jnp.array([True])
            }
            
            # --- Step 6: LR offsets ---
            # NOTE: We unconditionally compute LR offsets (no Python if-statement).
            # This is critical because Python conditionals inside JIT are evaluated at
            # trace-time, not runtime, which would "bake in" the branch from the first call.
            # 
            # This approach is safe because:
            # 1. Empty arrays (n_edges_lr=0): JAX handles them correctly, result is empty
            # 2. Padded edges: valid_lr=False → INVALID_OFFSET → d_ij >> cutoff → 0 contribution
            final_offset_lr = compute_offsets(system.R, safe_idx_i_lr, safe_idx_j_lr, inv_cell, valid_lr)
            inputs['idx_i_lr'] = safe_idx_i_lr
            inputs['idx_j_lr'] = safe_idx_j_lr
            inputs['cell_offset_lr'] = final_offset_lr
            
            # --- Step 7: Run model ---
            return so3lr_calc(inputs)
        
        # in_axes: system batched, neighbors batched
        return jax.jit(jax.vmap(unified_model_single, in_axes=(self._system_in_axes, self._neighbors_in_axes)))

    def _run_vmapped_calculation(self, batched_system, batched_neighbors, n_batch):
        """Execute vmapped calculation with UNIFIED GPU dispatch.
        
        OPTIMIZED: Single JIT function handles topology prep + cell inversion + model.
        Eliminates 2 Python dispatch round-trips compared to previous 3-call design.
        """
        
        # Lazy initialization of unified JIT function
        if self._unified_model_jit is None:
            self._unified_model_jit = self._create_unified_model_jit()
        
        # =====================================================================
        # SINGLE GPU DISPATCH: Topology + Inversion + Model all in one call
        # =====================================================================
        output_batched = self._unified_model_jit(batched_system, batched_neighbors)
        
        # Transfer ALL results to CPU in SINGLE device_get call (no separate block_until_ready needed)
        # device_get already blocks until computation is complete
        has_stress = 'stress' in output_batched
        if has_stress:
            energies_ev, forces_ev_ang, stresses_raw = jax.device_get(
                (output_batched['energy'], output_batched['forces'], output_batched['stress'])
            )
        else:
            energies_ev, forces_ev_ang = jax.device_get(
                (output_batched['energy'], output_batched['forces'])
            )
        
        # Vectorized unit conversion (fast numpy ops)
        energies_hartree = np.asarray(energies_ev) * EV_TO_HARTREE
        forces_hartree_bohr = np.asarray(forces_ev_ang) * (EV_TO_HARTREE * BOHR_TO_ANG)

        # Stress handling (optimized: no second device_get)
        if has_stress:
            stresses_raw = np.asarray(stresses_raw)
            if stresses_raw.shape[-1] == 6:
                # Vectorized Voigt to full tensor conversion
                stresses = np.zeros((n_batch, 3, 3), dtype=stresses_raw.dtype)
                stresses[:, 0, 0] = stresses_raw[:, 0]
                stresses[:, 1, 1] = stresses_raw[:, 1]
                stresses[:, 2, 2] = stresses_raw[:, 2]
                stresses[:, 1, 2] = stresses[:, 2, 1] = stresses_raw[:, 3]
                stresses[:, 0, 2] = stresses[:, 2, 0] = stresses_raw[:, 4]
                stresses[:, 0, 1] = stresses[:, 1, 0] = stresses_raw[:, 5]
            else:
                stresses = stresses_raw
            stresses = stresses * EV_TO_HARTREE * (BOHR_TO_ANG ** 3)
        else:
            stresses = self._zero_stresses  # Pre-allocated

        # Package results using pre-cached empty JSON
        result_list = [
            (float(energies_hartree[i]), forces_hartree_bohr[i].ravel(), stresses[i], self._empty_json)
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

        # 1. Unit Conversion
        cell_ang_b = np.asarray(cell_list) * BOHR_TO_ANG
        pos_ang_b = np.asarray(pos_list) * BOHR_TO_ANG

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
            def batch_leaf(x):
                if hasattr(x, 'ndim'):
                    return jnp.broadcast_to(x, (n_batch,) + x.shape)
                return x
            batched_system = jax.tree_util.tree_map(batch_leaf, self._system_template)
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

        if diagnostics and can_reuse_cache:
            print("[SO3LR] ✓ Fast Path: Reusing cached static inputs")

        # 7. Execute with GPU topology update
        results = self._run_vmapped_calculation(batched_system, batched_neighbors, n_batch)

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
