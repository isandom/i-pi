"""SO3LR driver with End-to-End JIT optimization.

This driver implements the "Full JIT" strategy:
1. Fuses NL update + graph construction + model evaluation into a SINGLE jax.jit function.
2. Uses vectorized tensor operations instead of Python loops for graph packing.
3. Never leaves the GPU during the hot path (only input/output transfers).
4. Computes cell offsets inside JIT using the robust formula: offset = -round(disp @ inv_cell).

Performance Architecture:
========================
BEFORE (3 separate stages with CPU bottleneck):
  JIT(NL) → CPU(Python loops, jraph.batch) → JIT(Model)

AFTER (single fused JIT):
  CPU(unit conversion) → JIT(NL + Graph + Model) → CPU(format output)

This eliminates:
- 2+ GPU/CPU synchronization points per step
- Python loop overhead for graph construction  
- jraph.dynamically_batch CPU overhead
- PCI-E data transfer penalties

Expected speedup: 20-40% reduction in wall-clock time per step.
"""

import os
import json
import time
import logging
import pathlib

# Configure JAX memory management BEFORE importing JAX
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np

from ase import Atoms
from ase.io import read


__DRIVER_NAME__ = "so3lr_jraph_main"
__DRIVER_CLASS__ = "SO3LR_driver"

# --- Unit conversion constants ---
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV


class SO3LR_driver(object):
    """
    SO3LR driver with End-to-End JIT optimization.
    
    Fuses neighbor list update, graph construction, and model evaluation
    into a single JIT-compiled function for maximum GPU efficiency.
    """

    def __init__(self, verbose=False, *args, **kwargs):
        self.verbose = verbose
        self.args = args
        self.kwargs = kwargs
        
        # Calculators
        self.calculator = None  # mlffCalculatorSparse (for initialization only)
        self.so3lr_calc = None  # So3lr model
        
        # Template and cached objects
        self.template_atoms = None
        self.atomic_numbers = None
        self.n_atoms = 0
        self._atomic_numbers_jax = None  # JAX array for use inside JIT
        
        # GLP neighbor list functions
        self._glp_allocate_fn = None
        self._glp_update_fn = None
        
        # End-to-end JIT function (THE core of this driver)
        self._end_to_end_fn = None
        self._neighbor_cache = None  # Stacked template neighbors for batch
        
        # Capacity tracking (for stable JIT shapes)
        self._max_sr_capacity = 0
        self._max_lr_capacity = 0
        self._n_batch_cached = 0
        
        # Cutoffs and parameters
        self.cutoff = None
        self.lr_cutoff = None
        self.dtype = None
        
        self.eval_count = 0
        
        # Pre-allocated CPU zeros for stress (i-PI interface requirement)
        self._stress_zeros = np.zeros((3, 3), dtype=np.float64)
        
        self.check_parameters()

    def check_parameters(self):
        """Initialize the SO3LR calculator and JIT machinery."""
        
        if self.verbose:
            print("[SO3LR-full-jit] Initializing driver...")
        
        # Enable JAX x64 mode BEFORE importing JAX if float64 requested
        dtype_str = self.kwargs.get('dtype', 'float32')
        if dtype_str == 'float64':
            os.environ['JAX_ENABLE_X64'] = 'true'
            if self.verbose:
                print("[SO3LR-full-jit] ⚠️  Enabling JAX x64 mode for float64 precision")
        
        # Import JAX and related libraries (AFTER x64 config)
        global jax, jnp
        
        import jax
        import jax.numpy as jnp
        
        from mlff.md import mlffCalculatorSparse
        from glp import atoms_to_system
        from glp.neighborlist import quadratic_neighbor_list
        from so3lr import So3lr
        
        # Store imports
        self.atoms_to_system = atoms_to_system
        self.quadratic_neighbor_list = quadratic_neighbor_list
        
        if self.verbose:
            print(f"[SO3LR-full-jit] JAX devices: {jax.devices()}")
            if dtype_str == 'float64':
                print(f"[SO3LR-full-jit] ✓ Float64 precision enabled (x64={jax.config.x64_enabled})")
        
        # Load template
        template_path = self.kwargs.get('template')
        if not template_path:
            raise ValueError("Must provide 'template' parameter with path to xyz file")
        
        self.template_atoms = read(template_path)
        self.template_atoms.set_pbc(True)
        self.atomic_numbers = self.template_atoms.get_atomic_numbers()
        self.n_atoms = len(self.template_atoms)
        
        if self.verbose:
            print(f"[SO3LR-full-jit] Loaded template with {self.n_atoms} atoms")
        
        # Get parameters
        self.lr_cutoff = float(self.kwargs.get('lr_cutoff', 12.0))
        self.cutoff = float(self.kwargs.get('cutoff', 4.5))
        self.dtype = np.float64 if dtype_str == 'float64' else np.float32
        
        # Neighbor list parameters
        self.capacity_multiplier = float(self.kwargs.get('capacity_multiplier', 1.25))
        
        # Find so3lr params directory
        model_path = self.kwargs.get('model_path')
        if model_path:
            params_dir = pathlib.Path(model_path)
        else:
            try:
                import so3lr as so3lr_pkg
                params_dir = pathlib.Path(so3lr_pkg.__file__).parent / 'params'
            except ImportError:
                raise ImportError("Could not find so3lr package. Provide 'model_path' parameter.")
        
        # Create mlffCalculatorSparse (only used for initialization reference)
        self.calculator = mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=params_dir,
            lr_cutoff=self.lr_cutoff,
            from_file=True,
            calculate_stress=False,
            dtype=self.dtype,
            skin=0.0,  # We always use force_update=True, so skin-based caching is disabled
            capacity_multiplier=self.capacity_multiplier,
        )
        
        # Create SO3LR calculator for evaluation (used inside JIT)
        self.so3lr_calc = So3lr(
            calculate_forces=True,
            lr_cutoff=self.lr_cutoff
        )
        
        if self.verbose:
            print("[SO3LR-full-jit] ✓ SO3LR calculator created")

    # =========================================================================
    # Initialization Methods
    # =========================================================================

    def _initialize_for_batch(self, positions_batched, cells_batched, n_batch):
        """Initialize GLP neighbor lists and create end-to-end JIT function.
        
        Called on first compute_batch or when batch size changes.
        """
        if self.verbose:
            print(f"[SO3LR-full-jit] Initializing for batch_size={n_batch}...")
        
        # Store atomic numbers as JAX array for use inside JIT
        self._atomic_numbers_jax = jnp.array(self.atomic_numbers, dtype=jnp.int32)
        
        # Initialize GLP neighbor list functions
        # Use first system's cell as template
        cell_init = cells_batched[0]
        pos_init = positions_batched[0]
        
        # skin=0.0: We always use force_update=True, so skin-based caching is disabled
        self._glp_allocate_fn, self._glp_update_fn = self.quadratic_neighbor_list(
            cell_init, 
            cutoff=self.cutoff, 
            skin=0.0,
            capacity_multiplier=self.capacity_multiplier, 
            lr_cutoff=self.lr_cutoff
        )
        
        # Allocate initial neighbor list
        neighbor_template = self._glp_allocate_fn(pos_init)
        
        # Track capacities for stable shapes
        self._max_sr_capacity = int(neighbor_template.centers.shape[0] * 1.25)
        if hasattr(neighbor_template, 'idx_i_lr') and neighbor_template.idx_i_lr is not None:
            self._max_lr_capacity = int(neighbor_template.idx_i_lr.shape[0] * 1.25)
        else:
            # Initialize with reasonable default for distant molecules
            self._max_lr_capacity = max(self.n_atoms * self.n_atoms // 4, 64)
        
        # Pad template to target capacity
        neighbor_template = self._pad_neighbor_list(
            neighbor_template, self._max_sr_capacity, self._max_lr_capacity
        )
        
        # Create stacked template for batch (broadcast single template to all batch items)
        self._neighbor_cache = jax.tree_util.tree_map(
            lambda x: self._broadcast_to_batch(x, n_batch), neighbor_template
        )
        
        self._n_batch_cached = n_batch
        
        # Create the end-to-end JIT function
        self._end_to_end_fn = self._create_end_to_end_jit(n_batch)
        
        if self.verbose:
            print(f"[SO3LR-full-jit] ✓ Initialized: SR_cap={self._max_sr_capacity}, LR_cap={self._max_lr_capacity}")
            print("[SO3LR-full-jit] ✓ End-to-end JIT function created")

    def _broadcast_to_batch(self, x, n_batch):
        """Broadcast array to include batch dimension."""
        if hasattr(x, 'ndim'):
            return jnp.broadcast_to(x, (n_batch,) + x.shape)
        return x

    def _pad_neighbor_list(self, neighbors, target_sr_cap, target_lr_cap):
        """Pad neighbor list arrays to target capacity for stable JIT shapes."""
        current_sr_cap = neighbors.centers.shape[0] if hasattr(neighbors, 'centers') else 0
        current_lr_cap = 0
        if hasattr(neighbors, 'idx_i_lr') and neighbors.idx_i_lr is not None:
            current_lr_cap = neighbors.idx_i_lr.shape[0]
        
        pad_sr = max(0, target_sr_cap - current_sr_cap)
        pad_lr = max(0, target_lr_cap - current_lr_cap)
        
        if pad_sr == 0 and pad_lr == 0:
            return neighbors
        
        def pad_array(arr):
            if not hasattr(arr, 'shape') or arr.ndim == 0:
                return arr
            
            # Pad SR arrays
            if arr.shape[0] == current_sr_cap and pad_sr > 0:
                pad_width = [(0, pad_sr)] + [(0, 0)] * (arr.ndim - 1)
                return jnp.pad(arr, pad_width, mode='constant', constant_values=0)
            
            # Pad LR arrays
            if target_lr_cap and arr.shape[0] == current_lr_cap and pad_lr > 0:
                pad_width = [(0, pad_lr)] + [(0, 0)] * (arr.ndim - 1)
                return jnp.pad(arr, pad_width, mode='constant', constant_values=0)
            
            return arr
        
        return jax.tree_util.tree_map(pad_array, neighbors)

    # =========================================================================
    # THE CORE: End-to-End JIT Function
    # =========================================================================

    def _create_end_to_end_jit(self, n_batch):
        """Create the fused JIT function: NL Update → Graph Construction → Model Eval.
        
        This is the heart of the optimization. Everything inside this function
        runs on the GPU without any CPU round-trips.
        """
        n_atoms = self.n_atoms
        atomic_numbers = self._atomic_numbers_jax
        so3lr_model = self.so3lr_calc
        glp_update_fn = self._glp_update_fn
        dtype = jnp.float64 if self.dtype == np.float64 else jnp.float32
        
        def end_to_end_compute(positions_batched, cells_batched, neighbor_cache):
            """
            End-to-end JIT-compiled compute function.
            
            Args:
                positions_batched: (B, N, 3) atom positions in Angstrom
                cells_batched: (B, 3, 3) cell matrices (row vectors)
                neighbor_cache: Stacked GLP neighbor lists from previous step
            
            Returns:
                energies: (B,) energies in eV
                forces: (B, N, 3) forces in eV/Angstrom
                updated_neighbors: Updated neighbor cache for next step
            """
            # For debugging JIT tracing
            # print("[DEBUG] TRACING END-TO-END JIT")
            
            batch_size = positions_batched.shape[0]
            
            # =================================================================
            # Stage 1: Neighbor List Update (vmapped GLP)
            # =================================================================
            def update_single_nl(pos, nbrs, cell):
                """Update neighbor list for single system."""
                return glp_update_fn(pos, nbrs, new_cell=cell, force_update=True)
            
            # Compute in_axes for neighbor cache (all leaves have batch dim 0)
            neighbor_in_axes = jax.tree_util.tree_map(lambda _: 0, neighbor_cache)
            
            # Vmapped update
            updated_neighbors = jax.vmap(
                update_single_nl,
                in_axes=(0, neighbor_in_axes, 0)
            )(positions_batched, neighbor_cache, cells_batched)
            
            # =================================================================
            # Stage 2: Vectorized Graph Construction (The Shift Trick)
            # =================================================================
            # 
            # IMPORTANT: We use jnp.where instead of boolean indexing to keep
            # shapes fixed. This is required because JAX JIT needs static shapes.
            # Invalid edges become self-loops (i→i) which have zero displacement
            # and thus zero contribution to forces/energy.
            #
            sr_cap = updated_neighbors.centers.shape[1]
            lr_cap = updated_neighbors.idx_i_lr.shape[1]
            
            # Total nodes: B*N real nodes + 1 padding node (like jraph.dynamically_batch)
            n_real_nodes = batch_size * n_atoms
            PADDING_NODE_IDX = n_real_nodes  # Index of the padding node
            n_total_nodes = n_real_nodes + 1
            
            # =================================================================
            # JRAPH-STYLE PADDING NODE MECHANISM
            # =================================================================
            # 
            # Instead of using edge_mask (which requires mlff patches), we add
            # a PADDING NODE with atomic_number=0 at the end of the node arrays.
            # Invalid edges point to this padding node.
            # 
            # This replicates what jraph.dynamically_batch does:
            # - init_masks() computes: point_mask = (z != 0)
            # - Padding node has z=0 → point_mask[PADDING_NODE_IDX] = 0
            # - Any contribution from/to padding node is automatically zeroed
            # - NO mlff patches required!
            #
            
            # --- Node Features: (B, N, ...) → (B*N + 1, ...) ---
            # Add one padding node with position=0, z=0, etc.
            positions_real = positions_batched.reshape(-1, 3)
            positions_flat = jnp.concatenate([
                positions_real,
                jnp.zeros((1, 3), dtype=dtype)  # Padding node at origin
            ])
            
            atomic_numbers_flat = jnp.concatenate([
                jnp.tile(atomic_numbers, batch_size),
                jnp.array([0], dtype=jnp.int32)  # z=0 → point_mask=0 automatically!
            ])
            
            # Cell per atom: (B*N + 1, 3, 3), padding node gets first cell (arbitrary)
            cell_per_atom = jnp.concatenate([
                jnp.repeat(cells_batched, n_atoms, axis=0),
                cells_batched[0:1]  # Padding node uses first cell (won't matter)
            ])
            
            # Batch metadata
            # Padding node belongs to a "padding graph" (index = batch_size)
            batch_segments = jnp.concatenate([
                jnp.repeat(jnp.arange(batch_size), n_atoms),
                jnp.array([batch_size], dtype=jnp.int32)  # Padding graph
            ])
            
            # Node mask: all real nodes = 1.0, padding node = 0.0
            node_mask = jnp.concatenate([
                jnp.ones(n_real_nodes, dtype=dtype),
                jnp.zeros(1, dtype=dtype)  # Padding node masked
            ])
            
            # Other node features
            forces_flat = jnp.zeros((n_total_nodes, 3), dtype=dtype)
            hirshfeld_flat = jnp.zeros(n_total_nodes, dtype=dtype)
            
            # --- Short-Range Edge Indices with Batch Offset ---
            batch_offsets = jnp.arange(batch_size) * n_atoms  # [0, N, 2N, ...]
            
            # Flatten indices: (B, Cap) → (B*Cap,)
            idx_i_flat = updated_neighbors.centers.reshape(-1)
            idx_j_flat = updated_neighbors.others.reshape(-1)
            
            # Create matching offsets
            sr_offsets = jnp.repeat(batch_offsets, sr_cap)
            
            # Validity mask: valid if indices < n_atoms and not self-loop
            valid_sr_mask = (updated_neighbors.centers < n_atoms).reshape(-1)
            valid_sr_mask = valid_sr_mask & (updated_neighbors.centers != updated_neighbors.others).reshape(-1)
            
            # Apply batch offset to get global indices
            idx_i_global = idx_i_flat + sr_offsets
            idx_j_global = idx_j_flat + sr_offsets
            
            # JRAPH-STYLE: Invalid edges → point to PADDING NODE (not self-loop on node 0)
            idx_i_sr = jnp.where(valid_sr_mask, idx_i_global, PADDING_NODE_IDX)
            idx_j_sr = jnp.where(valid_sr_mask, idx_j_global, PADDING_NODE_IDX)
            
            # --- Long-Range Edge Indices (Same Pattern) ---
            idx_i_lr_flat = updated_neighbors.idx_i_lr.reshape(-1)
            idx_j_lr_flat = updated_neighbors.idx_j_lr.reshape(-1)
            
            lr_offsets = jnp.repeat(batch_offsets, lr_cap)
            
            valid_lr_mask = (updated_neighbors.idx_i_lr < n_atoms).reshape(-1)
            valid_lr_mask = valid_lr_mask & (updated_neighbors.idx_i_lr != updated_neighbors.idx_j_lr).reshape(-1)
            
            idx_i_lr_global = idx_i_lr_flat + lr_offsets
            idx_j_lr_global = idx_j_lr_flat + lr_offsets
            
            # JRAPH-STYLE: Invalid LR edges → padding node
            idx_i_lr = jnp.where(valid_lr_mask, idx_i_lr_global, PADDING_NODE_IDX)
            idx_j_lr = jnp.where(valid_lr_mask, idx_j_lr_global, PADDING_NODE_IDX)
            
            # --- Compute Cell Offsets (OPTIMIZED: Fractional Positions) ---
            #
            # OPTIMIZATION: Instead of computing inv_cell per-edge (B*sr_cap + B*lr_cap inversions),
            # we compute it ONCE per bead (B inversions), then pre-compute fractional positions.
            # Cell offset = -round(frac[j] - frac[i]) uses only gather + subtract per edge.
            #
            # Cost reduction:
            #   - Inversions: B*(sr_cap + lr_cap) → B  (e.g., 576,000 → 32)
            #   - Matmuls: B*(sr_cap + lr_cap)*9 → B*N*9  (e.g., 5.2M → 47k FLOPs)
            #
            
            # Step 1: Compute inverse cells ONCE per bead (B inversions total)
            inv_cells_batched = jnp.linalg.inv(cells_batched)  # (B, 3, 3)
            
            # Step 2: Compute fractional positions for all atoms in batch
            # positions_batched: (B, N, 3), inv_cells_batched: (B, 3, 3)
            frac_positions_batched = jnp.einsum('bni,bij->bnj', positions_batched, inv_cells_batched)  # (B, N, 3)
            
            # Flatten to (B*N, 3) and add padding node at origin
            frac_positions_flat = jnp.concatenate([
                frac_positions_batched.reshape(-1, 3),
                jnp.zeros((1, 3), dtype=dtype)  # Padding node has frac coords (0,0,0)
            ])
            
            # Step 3: SR offsets - gather fractional coords, subtract, round
            frac_i_sr = frac_positions_flat[idx_i_sr]  # (B*sr_cap, 3)
            frac_j_sr = frac_positions_flat[idx_j_sr]  # (B*sr_cap, 3)
            cell_offset_sr = -jnp.round(frac_j_sr - frac_i_sr).astype(jnp.int32)
            
            # Step 4: LR offsets - same pattern
            frac_i_lr = frac_positions_flat[idx_i_lr]  # (B*lr_cap, 3)
            frac_j_lr = frac_positions_flat[idx_j_lr]  # (B*lr_cap, 3)
            cell_offset_lr = -jnp.round(frac_j_lr - frac_i_lr).astype(jnp.int32)
            
            # --- Edge Cells (needed by model for displacement computation) ---
            # This is separate from the cell offset optimization above - the model
            # still needs the cell matrix per SR edge for PBC displacement calculation.
            edge_cells = jnp.repeat(cells_batched, sr_cap, axis=0)  # (B*sr_cap, 3, 3)
            
            # --- Construct Inputs Dict ---
            # 
            # JRAPH-STYLE SAFETY: With the padding node mechanism, we don't need
            # edge_mask patches in mlff. The native init_masks() will compute:
            #   point_mask = (z != 0) → padding node gets point_mask = 0
            # Any contribution from/to the padding node is automatically zeroed.
            #
            # We still include edge_mask for compatibility, but it's not required
            # for correctness when using the padding node approach.
            #
            inputs = {
                # Node features (includes padding node)
                'positions': positions_flat,
                'atomic_numbers': atomic_numbers_flat,
                'cell_per_atom': cell_per_atom,
                'forces': forces_flat,
                'hirshfeld_ratios': hirshfeld_flat,
                
                # Batching metadata (padding node belongs to padding graph)
                'batch_segments': batch_segments,
                'node_mask': node_mask,
                
                # SR edge features (invalid edges point to padding node)
                'idx_i': idx_i_sr,
                'idx_j': idx_j_sr,
                'cell': edge_cells,
                'cell_offset': cell_offset_sr,
                
                # LR edge features (invalid edges point to padding node)
                'idx_i_lr': idx_i_lr,
                'idx_j_lr': idx_j_lr,
                'cell_offset_lr': cell_offset_lr,
                
                # Global features (includes padding graph)
                # Note: batch_size+1 graphs (real + padding), but we only extract first batch_size
                'energy': jnp.zeros(batch_size + 1, dtype=dtype),
                'total_charge': jnp.zeros(batch_size + 1, dtype=jnp.int16),
                'num_unpaired_electrons': jnp.zeros(batch_size + 1, dtype=jnp.int16),
                'theory_level': jnp.ones(batch_size + 1, dtype=jnp.int32),
                'theory_mask': jnp.tile(jnp.eye(16, dtype=jnp.float32)[1:2], (batch_size + 1, 1)),
                'graph_mask': jnp.concatenate([
                    jnp.ones(batch_size, dtype=bool),  # Real graphs
                    jnp.zeros(1, dtype=bool)  # Padding graph masked
                ]),
                'dipole_vec': jnp.zeros((batch_size + 1, 3), dtype=jnp.float32),
                'stress': jnp.zeros((batch_size + 1, 6), dtype=jnp.float32),
            }
            
            # =================================================================
            # Stage 3: Model Evaluation (SO3LR)
            # =================================================================
            output = so3lr_model(inputs)
            
            # Extract results (exclude padding graph/node)
            # Model returns batch_size+1 energies (includes padding graph energy)
            # Model returns n_total_nodes forces (includes padding node forces)
            all_energies = output['energy']
            all_forces = output['forces']
            
            # Only take first batch_size energies (ignore padding graph)
            energies = all_energies[:batch_size]
            
            # Only take first n_real_nodes forces (ignore padding node)
            # Then reshape to (B, N, 3)
            forces = all_forces[:n_real_nodes].reshape(batch_size, n_atoms, 3)
            
            return energies, forces, updated_neighbors
        
        # Compile the entire function
        return jax.jit(end_to_end_compute)

    # =========================================================================
    # Main Computation Methods
    # =========================================================================

    def compute_batch(self, cell_list, pos_list):
        """Main batch compute function.
        
        End-to-End JIT Architecture:
        ============================
        1. Unit conversion (CPU) - unavoidable
        2. Single JIT call (GPU) - NL + Graph + Model fused
        3. Result formatting (CPU) - unavoidable for i-PI interface
        
        No CPU/GPU round-trips in the hot path!
        """
        start = time.time()
        n_batch = len(pos_list)
        self.eval_count += 1
        show_diagnostics = (self.eval_count <= 3) or (self.eval_count % 50 == 0)
        
        if show_diagnostics:
            print(f"[SO3LR-full-jit] Batch #{self.eval_count}: {n_batch} structures")
        
        # =====================================================================
        # Step 1: Unit Conversion (CPU, unavoidable)
        # =====================================================================
        # Perform Bohr → Angstrom conversion in float64 for maximum precision
        cell_ang = (np.asarray(cell_list, dtype=np.float64) * BOHR_TO_ANG).astype(self.dtype)
        pos_ang = (np.asarray(pos_list, dtype=np.float64) * BOHR_TO_ANG).astype(self.dtype)
        
        # Transpose cells: i-PI uses columns, ASE/GLP use rows
        cells_jax = jnp.array(np.transpose(cell_ang, (0, 2, 1)))
        positions_jax = jnp.array(pos_ang)
        
        # =====================================================================
        # Step 2: Initialize if needed (first call or batch size change)
        # =====================================================================
        if self._end_to_end_fn is None or self._n_batch_cached != n_batch:
            self._initialize_for_batch(positions_jax, cells_jax, n_batch)
        
        # =====================================================================
        # Step 3: Single JIT Call (GPU) - THE HOT PATH
        # =====================================================================
        t_compute = time.time()
        
        energies, forces, self._neighbor_cache = self._end_to_end_fn(
            positions_jax, cells_jax, self._neighbor_cache
        )
        
        # Wait for completion and get results
        energies = jax.block_until_ready(energies)
        forces = jax.block_until_ready(forces)
        
        t_compute_time = time.time() - t_compute
        
        # =====================================================================
        # Step 4: Handle Overflow (rare, capacity exceeded)
        # =====================================================================
        overflow_flags = self._neighbor_cache.overflow
        any_overflow = bool(jax.device_get(jnp.any(overflow_flags)))
        
        if any_overflow:
            self._handle_overflow(positions_jax, cells_jax, n_batch)
            # Re-run with increased capacity
            energies, forces, self._neighbor_cache = self._end_to_end_fn(
                positions_jax, cells_jax, self._neighbor_cache
            )
            energies = jax.block_until_ready(energies)
            forces = jax.block_until_ready(forces)
        
        # =====================================================================
        # Step 5: Format Results (CPU, unavoidable for i-PI)
        # =====================================================================
        t_extract = time.time()
        
        # Convert to numpy with vectorized unit conversion (faster than per-item)
        energies_np = np.asarray(energies)
        forces_np = np.asarray(forces)
        
        # Vectorized unit conversion: eV → Hartree, eV/Å → Hartree/Bohr
        energies_hartree = energies_np * EV_TO_HARTREE
        forces_hartree = forces_np * (EV_TO_HARTREE * BOHR_TO_ANG)
        
        # Pre-create constant objects outside loop
        json_str = json.dumps({})
        stress_zeros = self._stress_zeros
        
        # Pack results (only reshape in loop, all math is done)
        result_list = [
            (energies_hartree[i].item(), forces_hartree[i].reshape(-1), stress_zeros, json_str)
            for i in range(n_batch)
        ]
        
        t_extract_time = time.time() - t_extract
        
        if show_diagnostics:
            t_total = time.time() - start
            print(f"[SO3LR-full-jit] JIT Compute: {t_compute_time*1000:.1f}ms, Extract: {t_extract_time*1000:.1f}ms")
            print(f"[SO3LR-full-jit] Total: {t_total:.3f}s ({t_total/n_batch:.4f}s/struct)")
        
        return result_list

    def _handle_overflow(self, positions_jax, cells_jax, n_batch):
        """Handle neighbor list overflow by growing capacity and reinitializing."""
        print(f"[SO3LR-full-jit] ⚠️  Overflow detected, growing capacity...")
        
        # Reallocate with larger capacity using first system
        new_neighbors = self._glp_allocate_fn(positions_jax[0], new_cell=cells_jax[0])
        
        # Update capacities with bucket rounding to minimize recompilations
        SR_BUCKET = 500
        LR_BUCKET = 1000
        
        def round_up_bucket(value, bucket):
            buffered = int(value * 1.25)
            return ((buffered + bucket - 1) // bucket) * bucket
        
        new_sr_cap = new_neighbors.centers.shape[0]
        new_lr_cap = new_neighbors.idx_i_lr.shape[0] if hasattr(new_neighbors, 'idx_i_lr') else 0
        
        self._max_sr_capacity = max(self._max_sr_capacity, round_up_bucket(new_sr_cap, SR_BUCKET))
        self._max_lr_capacity = max(self._max_lr_capacity, round_up_bucket(new_lr_cap, LR_BUCKET))
        
        print(f"[SO3LR-full-jit] Capacity grown: SR={self._max_sr_capacity}, LR={self._max_lr_capacity}")
        
        # Reinitialize with new capacity
        self._end_to_end_fn = None
        self._initialize_for_batch(positions_jax, cells_jax, n_batch)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def compute_structure(self, cell, pos):
        """Compute single structure."""
        return self.compute_batch([cell], [pos])[0]

    def compute(self, cell, pos):
        """Unified compute interface."""
        if isinstance(pos, list):
            return self.compute_batch(cell, pos)
        else:
            return self.compute_structure(cell, pos)

    def __call__(self, cell, pos):
        """Make driver callable."""
        return self.compute(cell, pos)
