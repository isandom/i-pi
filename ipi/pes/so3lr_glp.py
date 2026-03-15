"""SO3LR driver for i-PI using JAX vmap and native graph padding.
Evaluates PIMD beads efficiently and computes NVT forces or NPT stress tensors seamlessly using MLFF model endpoints natively via GLP neighbor list constructs.
"""

import os
import json
import time
import pathlib
import numpy as np
from ase.io import read

# Configure JAX memory management BEFORE importing JAX
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# Disable TF32 at CUDA driver level before importing JAX/CUDA runtime.
os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')

import jax
import jax.numpy as jnp

from ase.units import Bohr, Angstrom, Hartree, eV

# Fields that use LR capacity (explicit list to avoid size-based ambiguity)

_LR_FIELDS = frozenset({'idx_i_lr', 'idx_j_lr', 'cell_offset_lr', 'distance_lr'})

# Per-ATOM fields that should NOT be padded by edge capacity.
_ATOM_FIELDS = frozenset({'reference_positions', 'positions'})


def _build_glp_fused_kernel(
    *,
    calculator_fn,
    system_ctor,
    z_const,
    neighbors_in_axes,
    bohr_to_ang,
    ev_to_hartree,
    ev_ang_to_hartree_bohr,
    dtype,
    vacuum,
):
    """Build a pure fused GLP kernel (neighbor update + model compute)."""
    zero_stress = jnp.zeros((3, 3), dtype=dtype)
    def _convert_outputs(energies, forces, virials, updated_nbrs):
        return (
            energies * ev_to_hartree,
            forces * ev_ang_to_hartree_bohr,
            virials * ev_to_hartree,
            updated_nbrs,
            jnp.any(updated_nbrs.overflow),
        )

    if vacuum:
        def compute_single(neighbors, positions):
            system = system_ctor(R=positions, Z=z_const, cell=None)
            output, updated_nbrs = calculator_fn(system, neighbors)
            return output["energy"], output["forces"], output.get("stress", zero_stress), updated_nbrs

        vmapped_fn = jax.vmap(compute_single, in_axes=(neighbors_in_axes, 0))

        def fused_kernel(template_nbrs, positions_bohr):
            positions = positions_bohr * bohr_to_ang
            energies, forces, virials, updated_nbrs = vmapped_fn(template_nbrs, positions)
            return _convert_outputs(energies, forces, virials, updated_nbrs)
    else:
        def compute_single(neighbors, positions, cell):
            system = system_ctor(R=positions, Z=z_const, cell=cell)
            output, updated_nbrs = calculator_fn(system, neighbors)
            return output["energy"], output["forces"], output.get("stress", zero_stress), updated_nbrs

        vmapped_fn = jax.vmap(compute_single, in_axes=(neighbors_in_axes, 0, 0))

        def fused_kernel(template_nbrs, positions_bohr, cell_tensors_bohr):
            positions = positions_bohr * bohr_to_ang
            cells = jnp.transpose(cell_tensors_bohr * bohr_to_ang, (0, 2, 1))
            energies, forces, virials, updated_nbrs = vmapped_fn(template_nbrs, positions, cells)
            return _convert_outputs(energies, forces, virials, updated_nbrs)

    return jax.jit(fused_kernel, donate_argnums=(0,))


__DRIVER_NAME__ = "so3lr_glp"
__DRIVER_CLASS__ = "SO3LR_driver"


class SO3LR_driver(object):
    """
    SO3LR hardware-accelerated force field driver for i-PI.
    """

    def __init__(self, verbose=False, *args, **kwargs):
        self.verbose = bool(verbose)

        # Components
        self.so3lr_calc = None  # Direct So3lr model

        # Configuration
        self.vacuum = bool(kwargs.get('vacuum', False))
        self.calculate_stress = bool(kwargs.get('calculate_stress', False))
        self._template_path = kwargs.get('template')
        self.lr_cutoff = float(kwargs.get('lr_cutoff', 12.0))
        self.cutoff = float(kwargs.get('cutoff', 4.5))
        dtype_str = kwargs.get('dtype', 'float32')
        self.dtype = np.float32 if dtype_str == 'float32' else np.float64
        self.total_charge = int(kwargs.get('total_charge', 0))
        self.num_unpaired_electrons = int(kwargs.get('num_unpaired_electrons', 0))
        self.num_theory_levels = int(kwargs.get('num_theory_levels', 16))
        self.theory_level = int(kwargs.get('theory_level', 1))
        self.matmul_precision = kwargs.get('matmul_precision', 'highest')
        self.capacity_multiplier = self._resolve_capacity_multiplier(kwargs)
        self._model_path = kwargs.get('model_path')

        # Template and data
        self.template_atoms = None
        self.atomic_numbers = None
        self.n_atoms = 0
        self._system_template = None

        # Neighbor list machinery
        self._nl_initialized = False
        self._neighbor_template = None
        self._neighbor_allocator = None

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

        # Ignore positional arguments for API compatibility with i-PI.
        if args and self.verbose:
            print(f"[SO3LR] Ignoring unexpected positional args: {args}")

        # Initialize
        self._setup_model_and_kernels()

    # =========================================================================
    # Initialization
    # =========================================================================

    def _setup_model_and_kernels(self):
        """Consume static config and initialize model/setup kernels."""
        self._configure_jax_precision()

        from glp import System, atoms_to_system
        import so3lr as so3lr_pkg

        # Store references
        self._atoms_to_system = atoms_to_system
        self._System = System
        
        if self.verbose:
            print(f"[SO3LR] JAX devices: {jax.devices()}")

        # Load template
        if not self._template_path:
            raise ValueError("Must provide 'template' parameter with path to xyz file")

        self.template_atoms = read(self._template_path)
        # Set PBC based on vacuum mode
        self.template_atoms.set_pbc(not self.vacuum)
        self.atomic_numbers = self.template_atoms.get_atomic_numbers()
        self.n_atoms = len(self.template_atoms)

        if self.verbose:
            pbc_status = "VACUUM (no PBC)" if self.vacuum else "PERIODIC (PBC enabled)"
            print(f"[SO3LR] Loaded template with {self.n_atoms} atoms, mode: {pbc_status}")

        # Model path
        if self._model_path:
            params_dir = pathlib.Path(self._model_path)
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

        if not (0 <= self.theory_level < self.num_theory_levels):
            raise ValueError(
                f"theory_level must be in [0, {self.num_theory_levels - 1}], got {self.theory_level}"
            )

        # Pre-compute theory_mask as one-hot float32
        self._theory_mask_const = jnp.eye(self.num_theory_levels, dtype=jnp.float32)[self.theory_level:self.theory_level+1]
        
        if self.verbose:
            print(f"[SO3LR] Using theory_level={self.theory_level}")
            print("[SO3LR] ✓ Initialized (using NATIVE PADDING - no mlff patches needed)")
            
        self._build_constant_inputs()

    def _configure_jax_precision(self):
        """Configure numerical precision knobs once at driver initialization."""
        if self.dtype == np.float64:
            jax.config.update("jax_enable_x64", True)
            if self.verbose:
                print("[SO3LR] Enabling JAX x64 mode for float64 precision")

        if self.matmul_precision != 'default':
            jax.config.update("jax_default_matmul_precision", self.matmul_precision)
            if self.verbose:
                print(f"[SO3LR] Matmul precision set to '{self.matmul_precision}'")

    @staticmethod
    def _resolve_capacity_multiplier(kwargs):
        """Resolve capacity aliases while preserving backward compatibility."""
        capacity_multiplier = float(kwargs.get('capacity_multiplier', 1.25))
        for alias in ('buffer_size_multiplier', 'buffer_size_multiplier_sr', 'buffer_size_multiplier_lr'):
            if alias in kwargs:
                capacity_multiplier = float(kwargs[alias])
        return capacity_multiplier

    @staticmethod
    def _stack_leaf_broadcast(leaf, n_batch):
        """Broadcast one neighbor-list leaf to a fixed batch size."""
        if hasattr(leaf, 'ndim'):
            return jnp.broadcast_to(leaf, (n_batch,) + leaf.shape)
        return leaf

    def _build_constant_inputs(self):
        """Pre-allocate constant JAX arrays required by SO3LR and GLP."""
        self._model_constants = {
            'atomic_numbers': jnp.array(self.atomic_numbers, dtype=jnp.int32),
            'node_mask': jnp.ones(self.n_atoms, dtype=self.dtype),
            'hirshfeld_ratios': jnp.zeros(self.n_atoms, dtype=self.dtype),
            'forces': jnp.zeros((self.n_atoms, 3), dtype=self.dtype),
            'batch_segments': jnp.zeros(self.n_atoms, dtype=jnp.int32),
            'total_charge': jnp.array([self.total_charge], dtype=self.dtype),
            'num_unpaired_electrons': jnp.array([self.num_unpaired_electrons], dtype=self.dtype),
            'theory_mask': self._theory_mask_const,
            'graph_mask': jnp.array([True]),
            'energy': jnp.zeros(1, dtype=self.dtype),
        }
        
        self._unit_constants = {
            'bohr_to_ang': jnp.array(Bohr / Angstrom, dtype=self.dtype),
            'ev_to_hartree': jnp.array(eV / Hartree, dtype=self.dtype),
            'ev_ang_to_hartree_bohr': jnp.array((eV / Hartree) * (Bohr / Angstrom), dtype=self.dtype),
        }

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

    def _prepare_system_template(self, pos_ang_b, cell_ang_b=None):
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
                if cell_ang_b is None:
                    raise ValueError("Periodic mode requires a valid cell tensor")
                atoms.set_cell(cell_ang_b[0].T, scale_atoms=False)
                
            self._system_template = self._atoms_to_system(atoms, dtype=self.dtype)

        if not self._nl_initialized:
            from glp.neighborlist import quadratic_neighbor_list
            
            # CRITICAL: Pass cell=None for vacuum mode
            cell_init = None if self.vacuum else self._system_template.cell
            
            # skin=0.0: We always force a neighbor list update
            self._neighbor_allocator, _ = quadratic_neighbor_list(
                cell=cell_init, cutoff=self.cutoff, skin=0.0,
                capacity_multiplier=self.capacity_multiplier, lr_cutoff=self.lr_cutoff
            )
            positions_init = jnp.array(pos_ang_b[0], dtype=self.dtype)
            self._neighbor_template = self._neighbor_allocator(positions_init)
            

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
            
            self._compile_glp_model()
            self._nl_initialized = True
            
            if self.verbose:
                mode_str = "VACUUM (cell=None)" if self.vacuum else "PERIODIC"
                print(f"[SO3LR] GLP neighbor list initialized ({mode_str}, SR: {self._max_neighbor_capacity_seen}, LR: {self._max_neighbor_lr_capacity_seen})")

    def _compile_glp_model(self):
        """Construct the pure GLP calculator closures ahead of time."""
        def so3lr_energy_fn(graph):
            inputs = {
                **self._model_constants,
                'positions': graph.R,
                'cell_per_atom': None,
                'cell': getattr(graph, 'cell', None),
                'idx_i': graph.centers,
                'idx_j': graph.others,
                'edges': graph.edges,
                'idx_i_lr': graph.centers_lr,
                'idx_j_lr': graph.others_lr,
                'edges_lr': graph.edges_lr,
                'cell_lr': None,
            }
            output = self.so3lr_calc(inputs)
            e_tot = output['energy'][0]
            return jnp.full((self.n_atoms,), e_tot / self.n_atoms, dtype=self.dtype)

        from glp.potentials import Potential
        from glp.calculators import atom_pair_dual
        pot = Potential(so3lr_energy_fn, self.cutoff)
        pot.lr_cutoff = self.lr_cutoff
        
        calc, _ = atom_pair_dual.calculator(pot, self._system_template, skin=0.0, capacity_multiplier=self.capacity_multiplier)
        self._calculator_fn = calc.calculate

    def _ensure_stacked_template(self, n_batch, diagnostics=False):
        """Ensure stacked neighbor template and in_axes are ready for the batch size."""
        if (self._stacked_template_neighbors is None or 
            self._n_batch_cached != n_batch or
            self._template_version != id(self._neighbor_template)):
            
            self._stacked_template_neighbors = jax.tree_util.tree_map(
                lambda x: self._stack_leaf_broadcast(x, n_batch), self._neighbor_template
            )
            self._template_version = id(self._neighbor_template)
            self._neighbors_in_axes = self._compute_in_axes_for_neighbors(self._stacked_template_neighbors)
            
            self._compile_jit_kernel()
            self._n_batch_cached = n_batch
            
            if diagnostics:
                print(f"[SO3LR] Created stacked template for batch_size={n_batch}")

    def _compile_jit_kernel(self):
        """Build the batched jitted kernel based on the current template axes."""
        self._fused_update_and_compute = _build_glp_fused_kernel(
            calculator_fn=self._calculator_fn,
            system_ctor=self._System,
            z_const=self._model_constants['atomic_numbers'],
            neighbors_in_axes=self._neighbors_in_axes,
            bohr_to_ang=self._unit_constants['bohr_to_ang'],
            ev_to_hartree=self._unit_constants['ev_to_hartree'],
            ev_ang_to_hartree_bohr=self._unit_constants['ev_ang_to_hartree_bohr'],
            dtype=self.dtype,
            vacuum=self.vacuum,
        )

    def _handle_overflow(self, positions_bohr_batched, cell_tensors_bohr_batched, n_batch):
        """Handle neighbor list overflow by reallocating with increased capacity."""
        print("[SO3LR] Overflow detected, growing capacity")
        
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
            lambda x: self._stack_leaf_broadcast(x, n_batch), self._neighbor_template
        )
        # CRITICAL: recompute in_axes for the new neighbor list shapes
        self._neighbors_in_axes = self._compute_in_axes_for_neighbors(self._stacked_template_neighbors)
        # Force JIT kernel rebuild with new buffer shapes, and bypass _ensure_stacked_template's no-op guard
        self._fused_update_and_compute = None
        self._n_batch_cached = -1

    def _prepare_batch_inputs(self, cell_list, pos_list):
        """Convert host inputs to contiguous arrays and device buffers."""
        if pos_list is None:
            raise ValueError("pos_list cannot be None")

        n_batch = len(pos_list)
        if n_batch == 0:
            raise ValueError("Empty position batch is not supported")
        if cell_list is None:
            raise ValueError("cell_list cannot be None")
        if len(cell_list) != n_batch:
            raise ValueError(
                f"Batch size mismatch: len(cell_list)={len(cell_list)} vs len(pos_list)={n_batch}"
            )

        pos_arr = np.asarray(pos_list, dtype=np.float64)
        expected_shape = (n_batch, self.n_atoms, 3)
        if pos_arr.shape != expected_shape:
            raise ValueError(
                f"Expected positions with shape {expected_shape}, got {pos_arr.shape}"
            )
        positions_bohr_batched = jax.device_put(pos_arr.astype(self.dtype))

        if self.vacuum:
            return pos_arr, None, positions_bohr_batched, None

        cell_arr = np.asarray(cell_list, dtype=np.float64)
        expected_cell_shape = (n_batch, 3, 3)
        if cell_arr.shape != expected_cell_shape:
            raise ValueError(
                f"Expected cells with shape {expected_cell_shape}, got {cell_arr.shape}"
            )
        cell_tensors_bohr_batched = jax.device_put(cell_arr.astype(self.dtype))
        return pos_arr, cell_arr, positions_bohr_batched, cell_tensors_bohr_batched

    def _initialize_runtime_if_needed(self, pos_arr, cell_arr):
        """Initialize system and neighbor-list machinery on first evaluation."""
        if self._system_template is not None and self._nl_initialized:
            return

        pos_ang_b = (pos_arr * (Bohr / Angstrom)).astype(self.dtype)
        cell_ang_b = None
        if not self.vacuum:
            cell_ang_b = (cell_arr * (Bohr / Angstrom)).astype(self.dtype)
        self._prepare_system_template(pos_ang_b, cell_ang_b)

    def _format_results(self, energies_k, forces_k, virials_k):
        """Format one batch into i-PI expected tuple output."""
        n_batch = len(energies_k)
        return [
            (
                float(energies_k[i]),
                forces_k[i].ravel(),
                virials_k[i] if self.calculate_stress else self._zero_stresses[i],
                self._empty_json,
            )
            for i in range(n_batch)
        ]

    def _dispatch_fused(self, positions_bohr_batched, cell_tensors_bohr_batched):
        """Single dispatch of the fused NL-update + compute kernel.
        
        Returns:
            (energies, forces, overflow_flag)
            Also updates self._stacked_template_neighbors with the new NL state.
        """
        
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

        pos_arr, cell_arr, positions_bohr_batched, cell_tensors_bohr_batched = self._prepare_batch_inputs(
            cell_list, pos_list
        )

        # Initialize system/NL machinery (first call only)
        self._initialize_runtime_if_needed(pos_arr, cell_arr)
            
        # Ensure stacked template neighbors are ready
        self._ensure_stacked_template(n_batch, diagnostics)

        if diagnostics:
            t_convert = time.time() - t0
            t0 = time.time()

        # Single fused dispatch; overflow handled on a rare fallback path.
        energies, forces, virials, overflow = self._dispatch_fused(
            positions_bohr_batched, cell_tensors_bohr_batched
        )
        if bool(jax.device_get(overflow)):
            self._handle_overflow(positions_bohr_batched, cell_tensors_bohr_batched, n_batch)
            self._ensure_stacked_template(n_batch, diagnostics)
            energies, forces, virials, overflow = self._dispatch_fused(
                positions_bohr_batched, cell_tensors_bohr_batched
            )
            if bool(jax.device_get(overflow)):
                raise RuntimeError("Neighbor list overflow persists after reallocation")

        energies_k, forces_k, virials_k = jax.device_get(
            (energies, forces, virials)
        )
        results = self._format_results(energies_k, forces_k, virials_k)

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
