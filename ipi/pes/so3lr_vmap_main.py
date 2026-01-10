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
"""

import os
import json
import time
import pathlib
import numpy as np
from ase import Atoms
from ase.io import read


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

        # Neighbor list machinery
        self._nl_initialized = False
        self._neighbor_template = None
        self._neighbor_allocator = None
        self._glp_update_fn = None

        # Vmapped functions
        self._neighbors_in_axes = None
        
        # GPU JIT Function (fused topology + model computation with native padding)
        self._compute_jit = None

        # State / Cache
        self._batched_neighbors = None
        self._n_batch_cached = 0
        self._stacked_template_neighbors = None
        self._template_version = None
        
        # Fast Path cache
        self._cached_batched_system = None
        
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
        self.calc_stress = False  # Not yet supported
        self.damping = float(self.kwargs.get('dispersion_energy_cutoff_lr_damping', 2.0))
        self.total_charge = int(self.kwargs.get('total_charge', 0))
        self.num_unpaired_electrons = int(self.kwargs.get('num_unpaired_electrons', 0))

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
            atoms.set_cell(cell_ang_b[0].T, scale_atoms=False)
            self._system_template = self._atoms_to_system(atoms, dtype=self.dtype)

        if not self._nl_initialized:
            from glp.neighborlist import quadratic_neighbor_list
            
            cell_init = self._system_template.cell
            self._neighbor_allocator, self._glp_update_fn = quadratic_neighbor_list(
                cell=cell_init, cutoff=self.cutoff, skin=self.skin,
                capacity_multiplier=self.capacity_multiplier, lr_cutoff=self.lr_cutoff
            )
            positions_init = jnp.array(pos_ang_b[0], dtype=self.dtype)
            self._neighbor_template = self._neighbor_allocator(positions_init)
            
            initial_lr_estimate = max(self.n_atoms * self.n_atoms // 4, 32)
            self._max_neighbor_lr_capacity_seen = max(
                self._max_neighbor_lr_capacity_seen, 
                initial_lr_estimate
            )
            
            if hasattr(self._neighbor_template, 'centers'):
                initial_sr_capacity = int(self._neighbor_template.centers.shape[0] * 1.25)
                self._max_neighbor_capacity_seen = max(self._max_neighbor_capacity_seen, initial_sr_capacity)
            
            self._neighbor_template = self._pad_neighbor_list(
                self._neighbor_template,
                self._max_neighbor_capacity_seen,
                self._max_neighbor_lr_capacity_seen
            )
            
            self._nl_initialized = True
            
            if self.verbose:
                print(f"[SO3LR] ✓ GLP neighbor list initialized (SR: {self._max_neighbor_capacity_seen}, LR: {self._max_neighbor_lr_capacity_seen})")

    def _update_neighbor_lists(self, positions_batched, cell_tensors_batched, n_batch, diagnostics=False):
        """Update neighbor lists every step using force_update=True."""
        
        if (self._stacked_template_neighbors is None or 
            self._n_batch_cached != n_batch or
            self._template_version != id(self._neighbor_template)):
            
            self._stacked_template_neighbors = jax.tree_util.tree_map(
                lambda x: _stack_leaf_broadcast(x, n_batch), self._neighbor_template
            )
            self._template_version = id(self._neighbor_template)
            self._neighbors_in_axes = self._compute_in_axes_for_neighbors(self._stacked_template_neighbors)
            
            if diagnostics:
                print(f"[SO3LR] Created stacked template for batch_size={n_batch}")
        
        if not hasattr(self, '_vmapped_force_rebuild_fn') or self._vmapped_force_rebuild_fn is None:
            def force_rebuild_single(pos, nbrs, cell):
                return self._glp_update_fn(pos, nbrs, new_cell=cell, force_update=True)
            
            self._vmapped_force_rebuild_fn = jax.jit(jax.vmap(
                force_rebuild_single,
                in_axes=(0, self._neighbors_in_axes, 0)
            ))
        
        updated_neighbors_batched = self._vmapped_force_rebuild_fn(
            positions_batched,
            self._stacked_template_neighbors,
            cell_tensors_batched
        )
        
        # Check for overflow
        overflow_flags = updated_neighbors_batched.overflow
        any_overflow = bool(jax.device_get(jnp.any(overflow_flags)))
        
        if any_overflow:
            print(f"[SO3LR] ⚠️ Overflow detected, growing capacity")
            
            neighbors_list = self._unstack_pytree(updated_neighbors_batched, n_batch)
            overflow_np = np.asarray(overflow_flags)
            
            for i in range(n_batch):
                if overflow_np[i]:
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
                self._compute_jit = None
                self._vmapped_force_rebuild_fn = None
                self._stacked_template_neighbors = None
                print(f"[SO3LR] Capacity grown: SR={self._max_neighbor_capacity_seen}, LR={self._max_neighbor_lr_capacity_seen}")
            
            for i in range(len(neighbors_list)):
                neighbors_list[i] = self._pad_neighbor_list(
                    neighbors_list[i], self._max_neighbor_capacity_seen, self._max_neighbor_lr_capacity_seen
                )
            
            self._neighbor_template = jax.tree_util.tree_map(lambda x: x, neighbors_list[0])
            self._template_version = id(self._neighbor_template)
            updated_neighbors_batched = jax.tree.map(lambda *args: jnp.stack(args), *neighbors_list)
            
            self._stacked_template_neighbors = jax.tree_util.tree_map(
                lambda x: _stack_leaf_broadcast(x, n_batch), self._neighbor_template
            )
        
        self._batched_neighbors = updated_neighbors_batched
        self._n_batch_cached = n_batch
        
        return updated_neighbors_batched

    def _create_compute_jit(self):
        """Create FUSED JIT function with NATIVE PADDING mechanism.
        
        KEY DIFFERENCE FROM so3lr_vmap_main.py:
        ======================================
        Instead of using edge_mask (which requires mlff patches), we add a
        PADDING NODE with atomic_number=0 at the end of node arrays.
        
        Invalid edges are redirected to this padding node, not to node 0.
        
        Native init_masks() computes: point_mask = (z != 0)
        → Padding node gets point_mask = 0
        → Any contribution from/to padding node is automatically zeroed
        → NO mlff patches required!
        """
        dtype = self.dtype
        so3lr_calc = self.so3lr_calc
        n_atoms = self.n_atoms
        total_charge = self.total_charge
        num_unpaired_electrons = self.num_unpaired_electrons
        theory_mask_const = self._theory_mask_const
        atomic_numbers_const = jnp.array(self.atomic_numbers, dtype=jnp.int32)
        
        # PADDING NODE INDEX: One node after all real atoms
        PADDING_NODE_IDX = n_atoms
        n_total_nodes = n_atoms + 1
        
        def detect_valid_and_remap(idx_i, idx_j):
            """Detect valid edges and remap invalid ones to PADDING NODE (not node 0)."""
            valid_mask = (
                (idx_i >= 0) & (idx_j >= 0) & 
                (idx_i < n_atoms) & (idx_j < n_atoms) & 
                (idx_i != idx_j)
            )
            # NATIVE PADDING: Invalid edges point to padding node (z=0)
            safe_idx_i = jnp.where(valid_mask, idx_i, PADDING_NODE_IDX)
            safe_idx_j = jnp.where(valid_mask, idx_j, PADDING_NODE_IDX)
            return valid_mask, safe_idx_i, safe_idx_j
        
        def compute_offsets(frac_positions, safe_idx_i, safe_idx_j, valid_mask):
            """Compute integer cell offsets for PBC.
            
            For invalid edges (pointing to padding node), use zero offset.
            """
            frac_i = frac_positions[safe_idx_i]
            frac_j = frac_positions[safe_idx_j]
            disp_frac = frac_j - frac_i
            offset = -jnp.round(disp_frac).astype(jnp.int32)
            # Use zero offset for invalid edges (they point to padding node anyway)
            return jnp.where(valid_mask[:, None], offset, 0)
        
        def compute_single(neighbors, positions, cell):
            """Fused computation for a single bead with NATIVE PADDING.
            
            Adds a padding node with z=0 at the end of node arrays.
            Invalid edges point to this padding node.
            """
            # =====================================================
            # TOPOLOGY: Compute indices with NATIVE PADDING
            # =====================================================
            idx_i = neighbors.centers
            idx_j = neighbors.others
            
            # Remap invalid edges to PADDING NODE (not node 0)
            valid_sr, safe_idx_i, safe_idx_j = detect_valid_and_remap(idx_i, idx_j)
            
            # Handle LR indices
            idx_i_lr = getattr(neighbors, 'idx_i_lr', None)
            idx_j_lr = getattr(neighbors, 'idx_j_lr', None)
            
            if idx_i_lr is None:
                idx_i_lr = jnp.zeros((0,), dtype=jnp.int32)
            if idx_j_lr is None:
                idx_j_lr = jnp.zeros((0,), dtype=jnp.int32)
            
            valid_lr, safe_idx_i_lr, safe_idx_j_lr = detect_valid_and_remap(idx_i_lr, idx_j_lr)
            
            # =====================================================
            # CREATE NODE ARRAYS WITH PADDING NODE
            # =====================================================
            # Add one padding node with z=0 at the end
            
            # Positions: real atoms + padding node at origin
            positions_with_pad = jnp.concatenate([
                positions,
                jnp.zeros((1, 3), dtype=dtype)  # Padding node at origin
            ])
            
            # Atomic numbers: real atoms + z=0 for padding
            # z=0 → init_masks() computes point_mask=0 for this node automatically!
            atomic_numbers_with_pad = jnp.concatenate([
                atomic_numbers_const,
                jnp.array([0], dtype=jnp.int32)  # z=0 → point_mask=0
            ])
            
            # Cell per atom (for LR offset calculation)
            cell_per_atom = jnp.broadcast_to(cell[None, :, :], (n_total_nodes, 3, 3))
            
            # Node mask: 1.0 for real atoms, 0.0 for padding
            node_mask = jnp.concatenate([
                jnp.ones(n_atoms, dtype=dtype),
                jnp.zeros(1, dtype=dtype)
            ])
            
            # Batch segments: real atoms belong to graph 0, padding node to graph 1
            batch_segments = jnp.concatenate([
                jnp.zeros(n_atoms, dtype=jnp.int32),
                jnp.array([1], dtype=jnp.int32)  # Padding node → padding graph
            ])
            
            # Hirshfeld ratios (needed by model)
            hirshfeld_with_pad = jnp.zeros(n_total_nodes, dtype=dtype)
            
            # Forces placeholder
            forces_placeholder = jnp.zeros((n_total_nodes, 3), dtype=dtype)
            
            # =====================================================
            # COMPUTE CELL OFFSETS
            # =====================================================
            n_edges = safe_idx_i.shape[0]
            n_edges_lr = safe_idx_i_lr.shape[0]
            
            # Pre-compute fractional positions (O(n_atoms) matmul, done once)
            inv_cell = jnp.linalg.inv(cell)
            # Use positions_with_pad for fractional coords (includes padding node)
            frac_positions = jnp.dot(positions_with_pad, inv_cell)
            
            final_offset = compute_offsets(frac_positions, safe_idx_i, safe_idx_j, valid_sr)
            final_offset_lr = compute_offsets(frac_positions, safe_idx_i_lr, safe_idx_j_lr, valid_lr)
            
            # =====================================================
            # CONSTRUCT INPUTS with PADDING GRAPH
            # =====================================================
            # We have 2 graphs: real graph (index 0) + padding graph (index 1)
            
            inputs = {
                # Node features (includes padding node)
                'positions': positions_with_pad,
                'atomic_numbers': atomic_numbers_with_pad,
                'cell_per_atom': cell_per_atom,
                'node_mask': node_mask,
                'hirshfeld_ratios': hirshfeld_with_pad,
                'forces': forces_placeholder,
                
                # Batch metadata (2 graphs: real + padding)
                'batch_segments': batch_segments,
                
                # SR edge features (invalid edges point to padding node)
                'cell': jnp.broadcast_to(cell[None, :, :], (n_edges, 3, 3)),
                'idx_i': safe_idx_i,
                'idx_j': safe_idx_j,
                'cell_offset': final_offset,
                
                # LR edge features (invalid edges point to padding node)
                'idx_i_lr': safe_idx_i_lr,
                'idx_j_lr': safe_idx_j_lr,
                'cell_offset_lr': final_offset_lr,
                'cell_lr': jnp.broadcast_to(cell[None, :, :], (n_edges_lr, 3, 3)),
                
                # Global features (2 graphs)
                'total_charge': jnp.array([total_charge, 0], dtype=jnp.int16),
                'num_unpaired_electrons': jnp.array([num_unpaired_electrons, 0], dtype=jnp.int16),
                'theory_mask': jnp.tile(theory_mask_const, (2, 1)),  # 2 graphs
                'graph_mask': jnp.array([True, False]),  # Real graph True, padding graph False
                'energy': jnp.zeros(2, dtype=dtype),
            }

            # =====================================================
            # RUN MODEL
            # =====================================================
            output = so3lr_calc(inputs)
            
            # Extract results (ONLY from real graph/nodes)
            # Energy: only first graph (real), ignore padding graph
            energy = output['energy'][0]
            
            # Forces: only first n_atoms (real), ignore padding node
            forces = output['forces'][:n_atoms]
            
            # Unit conversion on GPU
            energy_hartree = energy * EV_TO_HARTREE
            forces_hartree_bohr = forces * (EV_TO_HARTREE * BOHR_TO_ANG)
            
            return energy_hartree, forces_hartree_bohr
        
        return jax.jit(jax.vmap(compute_single, in_axes=(self._neighbors_in_axes, 0, 0)))

    def _run_vmapped_calculation(self, batched_system, batched_neighbors, n_batch):
        """Execute vmapped calculation with NATIVE PADDING."""
        
        if self._compute_jit is None:
            self._compute_jit = self._create_compute_jit()
        
        energies_batched, forces_batched = self._compute_jit(
            batched_neighbors,
            batched_system.R,
            batched_system.cell
        )
        
        energies_k, forces_k = jax.device_get((energies_batched, forces_batched))
        
        result_list = [
            (float(energies_k[i]), forces_k[i].ravel(), self._zero_stresses[i], self._empty_json)
            for i in range(n_batch)
        ]

        return result_list

    # =========================================================================
    # Main Interface
    # =========================================================================

    def compute_batch(self, cell_list, pos_list):
        """Main batch compute function using vmap with NATIVE PADDING."""
        start = time.time()
        n_batch = len(pos_list)
        self.eval_count += 1
        diagnostics = (self.eval_count <= 3) or (self.eval_count % 50 == 0)

        if diagnostics:
            print(f"[SO3LR] Batch #{self.eval_count}: {n_batch} structures")
        
        # Pre-allocate zero stresses
        if self._zero_stresses is None or self._zero_stresses.shape[0] != n_batch:
            self._zero_stresses = np.zeros((n_batch, 3, 3), dtype=self.dtype)

        # Unit Conversion
        cell_ang_b = (np.asarray(cell_list, dtype=np.float64) * BOHR_TO_ANG).astype(self.dtype)
        pos_ang_b = (np.asarray(pos_list, dtype=np.float64) * BOHR_TO_ANG).astype(self.dtype)

        # Prepare JAX arrays
        positions_batched = jnp.array(pos_ang_b, dtype=self.dtype)
        cell_tensors_batched = jnp.array(np.transpose(cell_ang_b, (0, 2, 1)), dtype=self.dtype)

        # Initialize system/NL machinery
        self._prepare_system_template(pos_ang_b, cell_ang_b)

        # Create or reuse batched system
        if self._cached_batched_system is not None and n_batch == self._n_batch_cached:
            batched_system = self._cached_batched_system
        else:
            batched_system = jax.tree_util.tree_map(
                lambda x: _stack_leaf_broadcast(x, n_batch), self._system_template
            )
            if diagnostics:
                print(f"[SO3LR] Created batched system for batch_size={n_batch}")
        
        batched_system = batched_system._replace(
             R=positions_batched,
             cell=cell_tensors_batched
        )

        # Update neighbor lists
        self._update_neighbor_lists(
            positions_batched, cell_tensors_batched, n_batch, diagnostics
        )

        batched_neighbors = self._batched_neighbors

        # Execute fused computation with NATIVE PADDING
        results = self._run_vmapped_calculation(batched_system, batched_neighbors, n_batch)

        if diagnostics:
            t_total = time.time() - start
            print(f"[SO3LR] Total: {t_total:.3f}s ({t_total/n_batch:.4f}s/struct)")

        # Update cache
        self._cached_batched_system = batched_system
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
