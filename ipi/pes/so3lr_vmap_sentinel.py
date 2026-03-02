"""SO3LR driver for i-PI using JAX vmap and native graph padding.
Evaluates PIMD beads efficiently and computes NVT forces or NPT stress tensors seamlessly using MLFF model endpoints natively via GLP neighbor list constructs.
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


from ase.units import Bohr, Angstrom, Hartree, eV

from glp.system import System
from glp.calculators import atom_pair_dual

# Fields that use LR capacity (explicit list to avoid size-based ambiguity)

_LR_FIELDS = frozenset({'idx_i_lr', 'idx_j_lr', 'cell_offset_lr', 'distance_lr'})

# Per-ATOM fields that should NOT be padded by edge capacity.
_ATOM_FIELDS = frozenset({'reference_positions', 'positions'})





def _stack_leaf_broadcast(x, n_batch):
    """Broadcast array to include batch dimension."""
    if hasattr(x, 'ndim'):
        return jnp.broadcast_to(x, (n_batch,) + x.shape)
    return x


__DRIVER_NAME__ = "so3lr_vmap_sentinel"
__DRIVER_CLASS__ = "SO3LR_driver"


class SO3LR_driver(object):
    """
    SO3LR hardware-accelerated force field driver for i-PI.
    """

    def __init__(self, verbose=False, *args, **kwargs):
        self.verbose = verbose
        self.args = args
        self.kwargs = kwargs

        # Components
        self.so3lr_calc = None  # Direct So3lr model
        
        # Vacuum mode: disable PBC for isolated molecules
        self.vacuum = kwargs.get('vacuum', False)
        self.calculate_stress = kwargs.get('calculate_stress', False)

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
        self.damping = float(self.kwargs.get('dispersion_energy_cutoff_lr_damping', 2.0))
        self.total_charge = int(self.kwargs.get('total_charge', 0))
        self.num_unpaired_electrons = int(self.kwargs.get('num_unpaired_electrons', 0))

        # Neighbor list parameters (GLP only uses capacity_multiplier)
        self.capacity_multiplier = float(self.kwargs.get('capacity_multiplier', 1.25))
        for alias in ['buffer_size_multiplier', 'buffer_size_multiplier_sr', 'buffer_size_multiplier_lr']:
            if alias in self.kwargs:
                self.capacity_multiplier = float(self.kwargs[alias])

        # Model path
        model_path = self.kwargs.get('model_path')
        if model_path:
            params_dir = pathlib.Path(model_path)
        else:
            params_dir = pathlib.Path(so3lr_pkg.__file__).parent / 'params'
            if not params_dir.exists():
                raise ImportError(f"Could not find params at {params_dir}. Provide 'model_path'.")
        # Initialize SO3LR MLFF
        # We use calculate_forces=False because we compute them manually via value_and_grad on the energy sum
        try:
            from so3lr import So3lr
            self.so3lr_calc = So3lr(calculate_forces=False, lr_cutoff=self.lr_cutoff, neighborlist_format_lr='ordered_sparse')
        except Exception as e:
            raise ImportError(f"Failed to initialize So3lr calculator: {e}")

        # Theory level configuration
        self.num_theory_levels = int(self.kwargs.get('num_theory_levels', 16))
        self.theory_level = int(self.kwargs.get('theory_level', 1))
        
        # Pre-compute theory_mask as one-hot float32
        self._theory_mask_const = jnp.eye(self.num_theory_levels, dtype=jnp.float32)[self.theory_level:self.theory_level+1]
        
        if self.verbose:
            print(f"[SO3LR] Using theory_level={self.theory_level}")
            print("[SO3LR] ✓ Initialized (using NATIVE PADDING - no mlff patches needed)")

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
            cell_init = None if self.vacuum else self._system_template.cell
            
            # skin=0.0: We always use force_update=True
            from glp.periodic import make_displacement
            self._disp_fn = make_displacement(cell_init)
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

    def _handle_overflow(self, positions_bohr_batched, cell_tensors_bohr_batched, n_batch):
        """Handle neighbor list overflow by reallocating with increased capacity."""
        print(f"[SO3LR] ⚠️ Overflow detected, growing capacity")
        
        # Unstack and reallocate overflowed beads
        neighbors_list = self._unstack_pytree(self._batched_neighbors, n_batch)
        overflow_np = np.asarray(self._batched_neighbors.overflow)
        
        positions_ang_batched = np.asarray(positions_bohr_batched) * (Bohr / Angstrom)
        if not self.vacuum:
            cell_tensors_ang_batched = np.transpose(np.asarray(cell_tensors_bohr_batched) * (Bohr / Angstrom), (0, 2, 1))

        for i in range(n_batch):
            if overflow_np[i]:
                if self.vacuum:
                    neighbors_list[i] = self._neighbor_allocator(
                        positions_ang_batched[i], new_cell=None
                    )
                else:
                    neighbors_list[i] = self._neighbor_allocator(
                        positions_ang_batched[i], new_cell=cell_tensors_ang_batched[i]
                    )
        
        max_sr_cap = max(n.centers.shape[0] for n in neighbors_list)
        lr_caps = [n.idx_i_lr.shape[0] for n in neighbors_list
                   if hasattr(n, 'idx_i_lr') and n.idx_i_lr is not None]
        max_lr_cap = max(lr_caps) if lr_caps else 0
        
        old_sr_cap = self._max_neighbor_capacity_seen
        old_lr_cap = self._max_neighbor_lr_capacity_seen
        
        self._max_neighbor_capacity_seen = max(
            self._max_neighbor_capacity_seen, 
            int(max_sr_cap * self.capacity_multiplier)
        )
        self._max_neighbor_lr_capacity_seen = max(
            self._max_neighbor_lr_capacity_seen,
            int(max_lr_cap * self.capacity_multiplier)
        )
        
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
        # CRITICAL: recompute in_axes for the new neighbor list shapes
        self._neighbors_in_axes = self._compute_in_axes_for_neighbors(self._stacked_template_neighbors)
        # Force JIT kernel rebuild with new buffer shapes, and bypass _ensure_stacked_template's no-op guard
        self._fused_update_and_compute = None
        self._n_batch_cached = -1

    def _create_fused_update_and_compute(self):
        """Create a SINGLE fused JIT kernel: NL update + model compute."""
        dtype = self.dtype
        so3lr_calc = self.so3lr_calc
        n_atoms = self.n_atoms
        total_charge = self.total_charge
        num_unpaired_electrons = self.num_unpaired_electrons
        vacuum_mode = self.vacuum
        neighbors_in_axes = self._neighbors_in_axes
        
        # Constant hoisted arrays for So3lr inputs
        atomic_numbers_const = jnp.array(self.atomic_numbers, dtype=jnp.int32)
        node_mask_const = jnp.ones(n_atoms, dtype=dtype)
        hirshfeld_const = jnp.zeros(n_atoms, dtype=dtype)
        forces_placeholder_const = jnp.zeros((n_atoms, 3), dtype=dtype)
        total_charge_const = jnp.array([total_charge], dtype=dtype)
        num_unpaired_const = jnp.array([num_unpaired_electrons], dtype=dtype)
        theory_mask_const = self._theory_mask_const
        graph_mask_const = jnp.array([True])
        energy_placeholder_const = jnp.zeros(1, dtype=dtype)
        
        # This is strictly required because So3lr allows edges explicitly but still checks for positions shape 
        positions_placeholder_const = jnp.zeros((n_atoms, 3), dtype=dtype)

        def so3lr_energy_fn(graph):
            inputs = {
                'positions': graph.R,
                'atomic_numbers': atomic_numbers_const,
                'cell_per_atom': None,
                'node_mask': node_mask_const,
                'hirshfeld_ratios': hirshfeld_const,
                'forces': forces_placeholder_const,
                'batch_segments': jnp.zeros(n_atoms, dtype=jnp.int32),
                'cell': getattr(graph, 'cell', None),
                'idx_i': graph.centers,
                'idx_j': graph.others,
                'edges': graph.edges,
                'idx_i_lr': graph.centers_lr,
                'idx_j_lr': graph.others_lr,
                'edges_lr': graph.edges_lr,
                'cell_lr': None,
                'total_charge': total_charge_const,
                'num_unpaired_electrons': num_unpaired_const,
                'theory_mask': theory_mask_const,
                'graph_mask': graph_mask_const,
                'energy': energy_placeholder_const,
            }
            output = so3lr_calc(inputs)
            e_tot = output['energy'][0]
            energies = jnp.full((n_atoms,), e_tot / n_atoms, dtype=dtype)
            return energies

        from glp.potentials import Potential
        pot = Potential(so3lr_energy_fn, self.cutoff)
        pot.lr_cutoff = self.lr_cutoff
        
        # We manually instantiate GLP calculator instead of relying on `instantiate.py`
        # Because we already have the state from `self._neighbor_template` mapping, we just extract `calculator_fn`
        
        calc, _ = atom_pair_dual.calculator(pot, self._system_template, skin=0.0, capacity_multiplier=self.capacity_multiplier)
        calculator_fn = calc.calculate

        # Unit conversion scalars
        bohr_to_ang = jnp.array(Bohr / Angstrom, dtype=dtype)
        ev_to_hartree = jnp.array(eV / Hartree, dtype=dtype)
        ev_ang_to_hartree_bohr = jnp.array((eV / Hartree) * (Bohr / Angstrom), dtype=dtype)

        if vacuum_mode:
            def compute_single_vacuum(neighbors, positions):
                # We mock a System with Z=0 (since Z isn't really used when not using LJ)
                system = System(R=positions, Z=atomic_numbers_const, cell=None)
                output, updated_nbrs = calculator_fn(system, neighbors)
                energies = output["energy"]
                forces = output["forces"]
                stress = output.get("stress", jnp.zeros((3, 3), dtype=dtype))
                return energies, forces, stress, updated_nbrs

            vmapped_fn = jax.vmap(compute_single_vacuum, in_axes=(neighbors_in_axes, 0))

            def fused_kernel(template_nbrs, positions_bohr):
                positions = positions_bohr * bohr_to_ang
                energies, forces, virials, updated_nbrs = vmapped_fn(template_nbrs, positions)
                overflow = jnp.any(updated_nbrs.overflow)
                return energies * ev_to_hartree, forces * ev_ang_to_hartree_bohr, virials * ev_to_hartree, updated_nbrs, overflow

            return jax.jit(fused_kernel, donate_argnums=(0,))
        else:
            def compute_single(neighbors, positions, cell):
                system = System(R=positions, Z=atomic_numbers_const, cell=cell)
                output, updated_nbrs = calculator_fn(system, neighbors)
                energies = output["energy"]
                forces = output["forces"]
                stress = output.get("stress", jnp.zeros((3, 3), dtype=dtype))
                return energies, forces, stress, updated_nbrs

            vmapped_fn = jax.vmap(compute_single, in_axes=(neighbors_in_axes, 0, 0))

            def fused_kernel(template_nbrs, positions_bohr, cell_tensors_bohr):
                positions = positions_bohr * bohr_to_ang
                cells = jnp.transpose(cell_tensors_bohr * bohr_to_ang, (0, 2, 1))
                energies, forces, virials, updated_nbrs = vmapped_fn(template_nbrs, positions, cells)
                overflow = jnp.any(updated_nbrs.overflow)
                return energies * ev_to_hartree, forces * ev_ang_to_hartree_bohr, virials * ev_to_hartree, updated_nbrs, overflow

            return jax.jit(fused_kernel, donate_argnums=(0,))

    def _dispatch_fused(self, positions_bohr_batched, cell_tensors_bohr_batched):
        """Single dispatch of the fused NL-update + compute kernel.
        
        Returns:
            (energies, forces, overflow_flag)
            Also updates self._stacked_template_neighbors with the new NL state.
        """
        if self._fused_update_and_compute is None:
            self._fused_update_and_compute = self._create_fused_update_and_compute()
        
        if self.vacuum:
            energies, forces, virials, updated_nbrs, overflow = self._fused_update_and_compute(
                self._stacked_template_neighbors,
                positions_bohr_batched,
            )
        else:
            energies, forces, virials, updated_nbrs, overflow = self._fused_update_and_compute(
                self._stacked_template_neighbors,
                positions_bohr_batched,
                cell_tensors_bohr_batched,
            )
        
        # Store updated neighbors (donated input is now invalid, use output)
        self._stacked_template_neighbors = updated_nbrs
        self._batched_neighbors = updated_nbrs
        
        return energies, forces, virials, overflow

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

        # Type Conversion to GPU
        pos_arr = np.asarray(pos_list, dtype=np.float64)
        positions_bohr_batched = jax.device_put(pos_arr.astype(self.dtype))

        if self.vacuum:
            cell_arr = pos_arr
            cell_tensors_bohr_batched = None
        else:
            cell_arr = np.asarray(cell_list, dtype=np.float64)
            cell_tensors_bohr_batched = jax.device_put(cell_arr.astype(self.dtype))

        # Initialize system/NL machinery (first call only)
        if self._system_template is None:
            pos_ang_b = (pos_arr * (Bohr / Angstrom)).astype(self.dtype)
            if self.vacuum:
                cell_ang_b = pos_ang_b
            else:
                cell_ang_b = (cell_arr * (Bohr / Angstrom)).astype(self.dtype)
            self._prepare_system_template(pos_ang_b, cell_ang_b)
            
        # Ensure stacked template neighbors are ready
        self._ensure_stacked_template(n_batch, diagnostics)

        if diagnostics:
            t_convert = time.time() - t0
            t0 = time.time()

        # === FUSED DISPATCH: single JIT call does NL update + model compute ===
        energies, forces, virials, overflow = self._dispatch_fused(
            positions_bohr_batched, cell_tensors_bohr_batched
        )
        
        # Handle overflow (rare path — re-dispatch after reallocation)
        if bool(jax.device_get(overflow)):
            self._handle_overflow(positions_bohr_batched, cell_tensors_bohr_batched, n_batch)
            # Re-ensure template is valid after overflow reallocation
            self._ensure_stacked_template(n_batch, diagnostics)
            # Second dispatch with grown capacity
            energies, forces, virials, overflow = self._dispatch_fused(
                positions_bohr_batched, cell_tensors_bohr_batched
            )
            if bool(jax.device_get(overflow)):
                raise RuntimeError("Neighbor list overflow persists after reallocation")
        
        # Transfer results to CPU
        energies_k, forces_k, virials_k = jax.device_get((energies, forces, virials))
        
        results = [
            (float(energies_k[i]), forces_k[i].ravel(), virials_k[i] if self.calculate_stress else self._zero_stresses[i], self._empty_json)
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
