from src.boundary_conditions import *
from jax.config import config
from src.utils import *
from functools import partial
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from jax.sharding import PositionalSharding
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.experimental.multihost_utils import process_allgather
from jax import jit, lax, vmap
import time
import jax.numpy as jnp
import numpy as np
import jmp
import os
import jax

jax.config.update("jax_array", True)
jax.config.update("jax_spmd_mode", 'allow_all')
# Disables annoying TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class LBMBase(object):
    """
    LBMBase: A class that represents a base for Lattice Boltzmann Method simulation.
    
    Parameters
    ----------
        lattice (object): The lattice object that contains the lattice structure and weights.
        omega (float): The relaxation parameter for the LBM simulation.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        nz (int, optional): Number of grid points in the z-direction. Defaults to 0.
        precision (str, optional): A string specifying the precision used for the simulation. Defaults to "f32/f32".
        optimize (bool, optional): Whether or not to run adjoint optimization (not functional yet). Defaults to False.
    """
    
    def __init__(self, lattice, omega, nx, ny, nz=0, precision="f32/f32", optimize=False):
        # Set the precision for computation and storage
        computedType, storedType = self.set_precisions(precision)
        self.precision_policy = jmp.Policy(compute_dtype=computedType,
                                            param_dtype=computedType, output_dtype=storedType)
        self.precision = precision
        self.optimize = optimize
        self.n_devices = jax.device_count()

        # Check for distributed mode
        if self.n_devices > jax.local_device_count():
            print("WARNING: Running in distributed mode. Make sure that jax.distributed.initialize is called before performing any JAX computations.")
        print("XLA backend:", jax.default_backend())
        print("Number of XLA devices available:", self.n_devices)
        self.p_i = np.arange(self.n_devices)
        self.lattice = lattice
        self.omega = omega
        self.c = self.lattice.c
        self.q = self.lattice.q
        self.w = self.lattice.w
        self.dim = self.lattice.d
        self.ret_fpost = False
        """
        Adjust the number of grid points in the x direction, if necessary.
        If the number of grid points is not divisible by the number of devices
        it increases the number of grid points to the next multiple of the number of devices.
        This is done in order to accommodate the domain partitioning per XLA device
        """
        self.nx = nx
        if nx % self.n_devices:
            self.nx = nx + (self.n_devices - nx % self.n_devices)
            print("nx increased from {} to {} in order to accommodate the domain partitioning per XLA device.".format(nx, self.nx))

        self.ny = ny
        self.nz = nz
    
        # Store grid information
        self.grid_info = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dim": lattice.d,
            "lattice": lattice
        }

        P = PartitionSpec

        # Define the right permutation
        self.right_perm = [(i, (i + 1) % self.n_devices) for i in range(self.n_devices)]
        # Define the left permutation
        self.left_perm = [((i + 1) % self.n_devices, i) for i in range(self.n_devices)]


        # Set up the sharding and streaming for 2D and 3D simulations
        if self.dim == 2:
            self.devices = mesh_utils.create_device_mesh((self.n_devices, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "f"))
            self.namedSharding = NamedSharding(self.mesh, P("x", "y", "f"))

            self.positionalSharding = PositionalSharding(self.devices)
            self.streaming = jit(shard_map(self.streaming_m, mesh=self.mesh,
                                                      in_specs=P("x", None, None), out_specs=P("x", None, None), check_rep=False))


        # Set up the sharding and streaming for 2D and 3D simulations
        elif self.dim == 3:
            self.devices = mesh_utils.create_device_mesh((self.n_devices, 1, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "z", "f"))
            self.namedSharding = NamedSharding(self.mesh, P("x", "y", "z", "f"))

            self.positionalSharding = PositionalSharding(self.devices)
            self.streaming = jit(shard_map(self.streaming_m, mesh=self.mesh,
                                                      in_specs=P("x", None, None, None), out_specs=P("x", None, None, None), check_rep=False))
        else:
            raise ValueError(f"dim = {self.dim} not supported")
        
        # Compute the bounding box indices for boundary conditions
        self.boundingBoxIndices = self.bounding_box_indices()
        # Create boundary data for the simulation
        self._create_boundary_data()

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid connectivity bitmask, and preparing local bitmasks and normal arrays.
        """
        self.BCs = []
        self.set_boundary_conditions()
        # Accumulate the indices of all BCs to create the grid connectivity bitmask with FALSE along directions that
        # stream into a boundary voxel.
        solid_halo_list = [np.array(bc.indices).T for bc in self.BCs if bc.isSolid]
        solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

        # Create the grid connectivity bitmask on each process
        create_grid_connectivity_bitmask = jit((self.create_grid_connectivity_bitmask), backend='cpu')
        start = time.time()
        connectivity_bitmask = create_grid_connectivity_bitmask(solid_halo_voxels)
        print("Time to create the grid connectivity bitmask:", time.time() - start)

        start = time.time()
        for bc in self.BCs:
            assert bc.implementationStep in ['PostStreaming', 'PostCollision']
            bc.create_local_bitmask_and_normal_arrays(connectivity_bitmask)
        print("Time to create the local bitmasks and normal arrays:", time.time() - start)

    # This is another non-JITed way of creating the distributed arrays. It is not used at the moment.
    # def distributed_array_init(self, shape, type, initVal=None):
    #     sharding_dim = shape[0] // self.n_devices
    #     sharded_shape = (self.n_devices, sharding_dim,  *shape[1:])
    #     device_shape = sharded_shape[1:]
    #     arrays = []

    #     for d, index in self.namedSharding.addressable_devices_indices_map(sharded_shape).items():
    #         jax.default_device = d
    #         if initVal is None:
    #             x = jnp.zeros(shape=device_shape, dtype=type)
    #         else:
    #             x = jnp.full(shape=device_shape, fill_value=initVal, dtype=type)  
    #         arrays += [jax.device_put(x, d)] 
    #     jax.default_device = jax.devices()[0]
    #     return jax.make_array_from_single_device_arrays(shape, self.namedSharding, arrays)

    @partial(jit, static_argnums=(0, 1, 2, 4))
    def distributed_array_init(self, shape, type, initVal=0, sharding=None):
        """
        Initialize a distributed array using JAX, with a specified shape, data type, and initial value.
        Optionally, provide a custom sharding strategy.

        Parameters
        ----------
            shape (tuple): The shape of the array to be created.
            type (dtype): The data type of the array to be created.
            initVal (scalar, optional): The initial value to fill the array with. Defaults to 0.
            sharding (Sharding, optional): The sharding strategy to use. Defaults to `self.namedSharding`.

        Returns
        -------
            jax.numpy.ndarray: A JAX array with the specified shape, data type, initial value, and sharding strategy.
        """
        if sharding is None:
            sharding = self.namedSharding
        x = jnp.full(shape=shape, fill_value=initVal, dtype=type)        
        return jax.lax.with_sharding_constraint(x, sharding)
    
    @partial(jit, static_argnums=(0,))
    def create_grid_connectivity_bitmask(self, solid_halo_voxels):
        """
        This function creates a bitmask for the background grid that accounts for the location of the boundaries.
        
        Parameters
        ----------
            solid_halo_voxels: A numpy array representing the voxels in the halo of the solid object.
            
        Returns
        -------
            A JAX array representing the connectivity bitmask of the grid.
        """
        # Halo width (hw_x is different to accommodate the domain partitioning per XLA device)
        hw_x = self.n_devices
        hw_y = hw_z = 1
        if self.dim == 2:
            connectivity_bitmask = jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=bool)
            if solid_halo_voxels is not None:
                connectivity_bitmask = connectivity_bitmask.at[tuple(solid_halo_voxels.T)].set(True)
            connectivity_bitmask = jnp.pad(connectivity_bitmask, ((hw_x, hw_x), (hw_y, hw_y), (0, 0)),
                                           'constant', constant_values=True)
            connectivity_bitmask = self.compute_bitmask(connectivity_bitmask)

            return connectivity_bitmask[hw_x:-hw_x, hw_y:-hw_y]

        elif self.dim == 3:
            connectivity_bitmask = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=bool)
            if solid_halo_voxels is not None:
                connectivity_bitmask = connectivity_bitmask.at[tuple(solid_halo_voxels.T)].set(True)
            connectivity_bitmask = jnp.pad(connectivity_bitmask, ((hw_x, hw_x), (hw_y, hw_y), (hw_z, hw_z), (0, 0)),
                                           'constant', constant_values=True)
            connectivity_bitmask = self.compute_bitmask(connectivity_bitmask)

            return connectivity_bitmask[hw_x:-hw_x, hw_y:-hw_y, hw_z:-hw_z]

    def bounding_box_indices(self):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.

        Returns
        -------
            boundingBox (dict): A dictionary where keys are the names of the bounding box faces
            ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
            are numpy arrays of indices corresponding to each face.
        """
        if self.dim == 2:
            # For a 2D grid, the bounding box consists of four edges: bottom, top, left, and right.
            # Each edge is represented as an array of indices. For example, the bottom edge includes
            # all points where the y-coordinate is 0, so its indices are [[i, 0] for i in range(self.nx)].
            boundingBox = {"bottom": np.array([[i, 0] for i in range(self.nx)], dtype=int),
                           "top": np.array([[i, self.ny - 1] for i in range(self.nx)], dtype=int),
                           "left": np.array([[0, i] for i in range(self.ny)], dtype=int),
                           "right": np.array([[self.nx - 1, i] for i in range(self.ny)], dtype=int)}
                            
            return boundingBox

        elif self.dim == 3:
            # For a 3D grid, the bounding box consists of six faces: bottom, top, left, right, front, and back.
            # Each face is represented as an array of indices. For example, the bottom face includes all points
            # where the z-coordinate is 0, so its indices are [[i, j, 0] for i in range(self.nx) for j in range(self.ny)].
            boundingBox = {
                "bottom": np.array([[i, j, 0] for i in range(self.nx) for j in range(self.ny)], dtype=int),
                "top": np.array([[i, j, self.nz - 1] for i in range(self.nx) for j in range(self.ny)],dtype=int),
                "left": np.array([[0, j, k] for j in range(self.ny) for k in range(self.nz)], dtype=int),
                "right": np.array([[self.nx - 1, j, k] for j in range(self.ny) for k in range(self.nz)], dtype=int),
                "front": np.array([[i, 0, k] for i in range(self.nx) for k in range(self.nz)], dtype=int),
                "back": np.array([[i, self.ny - 1, k] for i in range(self.nx) for k in range(self.nz)], dtype=int)}

            return boundingBox

    def set_precisions(self, precision):
        """
        This function sets the precision of the computations. The precision is defined by a pair of values,
        representing the precision of the computation and the precision of the storage, respectively.

        Parameters
        ----------
            precision (str): A string representing the desired precision. The string should be in the format
            "computation/storage", where "computation" and "storage" are either "f64", "f32", or "f16",
            representing 64-bit, 32-bit, or 16-bit floating point numbers, respectively.

        Returns
        -------
            tuple: A pair of jax.numpy data types representing the computation and storage precisions, respectively.
            If the input string does not match any of the predefined options, the function defaults to (jnp.float32, jnp.float32).
        """
        return {
            "f64/f64": (jnp.float64, jnp.float64),
            "f32/f32": (jnp.float32, jnp.float32),
            "f32/f16": (jnp.float32, jnp.float16),
            "f16/f16": (jnp.float16, jnp.float16),
            "f64/f32": (jnp.float64, jnp.float32),
            "f64/f16": (jnp.float64, jnp.float16),
        }.get(precision, (jnp.float32, jnp.float32))

    def initialize_macroscopic_fields(self):
        """
        This function initializes the macroscopic fields (density and velocity) to their default values.
        The default density is 1 and the default velocity is 0.

        Note: This function is a placeholder and should be overridden in a subclass or in an instance of the class
        to provide specific initial conditions.

        Returns
        -------
            None, None: The default density and velocity, both None. This indicates that the actual values should be set elsewhere.
        """
        print("WARNING: Default initial conditions assumed: density = 1, velocity = 0")
        print("         To set explicit initial density and velocity, use self.initialize_macroscopic_fields.")
        return None, None

    def assign_fields_sharding(self):
        """
        This function is used to initialize the simulation by assigning the macroscopic fields and populations.

        The function first initializes the macroscopic fields, which are the density (rho0) and velocity (u0).
        Depending on the dimension of the simulation (2D or 3D), it then sets the shape of the array that will hold the 
        distribution functions (f).

        If the density or velocity are not provided, the function initializes the distribution functions with a default 
        value (self.w), representing density=1 and velocity=0. Otherwise, it uses the provided density and velocity to initialize the populations.

        Parameters
        ----------
        None

        Returns
        -------
        f: a distributed JAX array of shape (nx, ny, nz, q) or (nx, ny, q) holding the distribution functions for the simulation.
        """
        rho0, u0 = self.initialize_macroscopic_fields()

        if self.dim == 2:
            shape = (self.nx, self.ny, self.lattice.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.lattice.q)
    
        if rho0 is None or u0 is None:
            f = self.distributed_array_init(shape, self.precision_policy.output_dtype, initVal=self.w)
        else:
            f = self.initialize_populations(rho0, u0)

        return f
    
    def initialize_populations(self, rho0, u0):
        """
        This function initializes the populations (distribution functions) for the simulation.
        It uses the equilibrium distribution function, which is a function of the macroscopic 
        density and velocity.

        Parameters
        ----------
        rho0: jax.numpy.ndarray
            The initial density field.
        u0: jax.numpy.ndarray
            The initial velocity field.

        Returns
        -------
        f: jax.numpy.ndarray
            The array holding the initialized distribution functions for the simulation.
        """
        return self.equilibrium(rho0, u0)

    def send_right(self, x, axis_name):
        """
        This function sends the data to the right neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
        jax.numpy.ndarray
            The data after being sent to the right neighboring process.
        """
        return lax.ppermute(x, perm=self.right_perm, axis_name=axis_name)
   
    def send_left(self, x, axis_name):
        """
        This function sends the data to the left neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
            The data after being sent to the left neighboring process.
        """
        return lax.ppermute(x, perm=self.left_perm, axis_name=axis_name)
    
    def streaming_m(self, f):
        """
        This function performs the streaming step in the Lattice Boltzmann Method, which is 
        the propagation of the distribution functions in the lattice.

        To enable multi-GPU/TPU functionality, it extracts the left and right boundary slices of the
        distribution functions that need to be communicated to the neighboring processes.

        The function then sends the left boundary slice to the right neighboring process and the right 
        boundary slice to the left neighboring process. The received data is then set to the 
        corresponding indices in the receiving domain.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The array holding the distribution functions for the simulation.

        Returns
        -------
        jax.numpy.ndarray
            The distribution functions after the streaming operation.
        """
        f = self.streaming_p(f)
        left_comm, right_comm = f[:1, ..., self.lattice.right_indices], f[-1:, ..., self.lattice.left_indices]

        left_comm, right_comm = self.send_right(left_comm, 'x'), self.send_left(right_comm, 'x')
        f = f.at[:1, ..., self.lattice.right_indices].set(left_comm)
        f = f.at[-1:, ..., self.lattice.left_indices].set(right_comm)
        return f

    @partial(jit, static_argnums=(0,))
    def streaming_p(self, f):
        """
        Perform streaming operation on a partitioned (in the x-direction) distribution function.
        
        The function uses the vmap operation provided by the JAX library to vectorize the computation 
        over all lattice directions.

        Parameters
        ----------
            f: The distribution function.

        Returns
        -------
            The updated distribution function after streaming.
        """
        def streaming_i(f, c):
            """
            Perform individual streaming operation in a direction.

            Parameters
            ----------
                f: The distribution function.
                c: The streaming direction vector.

            Returns
            -------
                jax.numpy.ndarray
                The updated distribution function after streaming.
            """
            if self.dim == 2:
                return jnp.roll(f, (c[0], c[1]), axis=(0, 1))
            elif self.dim == 3:
                return jnp.roll(f, (c[0], c[1], c[2]), axis=(0, 1, 2))

        return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(f, self.c.T)

    def compute_bitmask(self, b):    
        """
        This function computes a bitmask for each direction in the lattice. The bitmask is used to 
        determine which nodes are fluid nodes and which are boundary nodes.

        It does this by rolling the input bitmask (b) in the opposite direction of each lattice 
        direction. The rolling operation shifts the values of the bitmask along the specified axes.

        The function uses the vmap operation provided by the JAX library to vectorize the computation 
        over all lattice directions.

        Parameters
        ----------
        b: ndarray
            The input bitmask.

        Returns
        -------
        jax.numpy.ndarray
            The computed bitmask for each direction in the lattice.
        """
        def compute_bitmask_i(b, i):
            """
            This function computes the bitmask for a specific direction in the lattice.

            It does this by rolling the input bitmask (b) in the opposite direction of the specified 
            lattice direction. The rolling operation shifts the values of the bitmask along the 
            specified axes.

            Parameters
            ----------
            b: jax.numpy.ndarray
                The input bitmask.
            i: int
                The index of the lattice direction.

            Returns
            -------
            jax.numpy.ndarray
                The computed bitmask for the specified direction in the lattice.
            """
            if self.dim == 2:
                rolls = (self.c.T[i, 0], self.c.T[i, 1])
                axes = (0, 1)
                return jnp.roll(b[..., self.lattice.opp_indices[i]], rolls, axes)
            elif self.dim == 3:
                rolls = (self.c.T[i, 0], self.c.T[i, 1], self.c.T[i, 2])
                axes = (0, 1, 2)
                return jnp.roll(b[..., self.lattice.opp_indices[i]], rolls, axes)

        return vmap(compute_bitmask_i, in_axes=(None, 0), out_axes=-1)(b, self.lattice.i_s)

    @partial(jit, static_argnums=(0, 3), inline=True)
    def equilibrium(self, rho, u, castOutput=True):
        """
        This function computes the equilibrium distribution function in the Lattice Boltzmann Method.
        The equilibrium distribution function is a function of the macroscopic density and velocity.

        The function first casts the density and velocity to the compute precision if the castOutput flag is True.
        The function finally casts the equilibrium distribution function to the output precision if the castOutput 
        flag is True.

        Parameters
        ----------
        rho: jax.numpy.ndarray
            The macroscopic density.
        u: jax.numpy.ndarray
            The macroscopic velocity.
        castOutput: bool, optional
            A flag indicating whether to cast the density, velocity, and equilibrium distribution function to the 
            compute and output precisions. Default is True.

        Returns
        -------
        feq: ja.numpy.ndarray
            The equilibrium distribution function.
        """
        # Cast the density and velocity to the compute precision if the castOutput flag is True
        if castOutput:
            rho, u = self.precision_policy.cast_to_compute((rho, u))

        # Cast c to compute precision so that XLA call FXX matmul, 
        # which is faster (it is faster in some older versions of JAX, newer versions are smart enough to do this automatically)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu = 3.0 * jnp.dot(u, c)
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        feq = rho[..., None] * self.w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

        if castOutput:
            return self.precision_policy.cast_to_output(feq)
        else:
            return feq

    @partial(jit, static_argnums=(0,))
    def momentum_flux(self, fneq):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium 
        distribution functions (fneq) and the lattice moments (cc).

        The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann 
        Method (LBM).

        Parameters
        ----------
        fneq: jax.numpy.ndarray
            The non-equilibrium distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The computed momentum flux.
        """
        return jnp.dot(fneq, self.lattice.cc)

    @partial(jit, static_argnums=(0,), inline=True)
    def update_macroscopic(self, f):
        """
        This function computes the macroscopic variables (density and velocity) based on the 
        distribution functions (f).

        The density is computed as the sum of the distribution functions over all lattice directions. 
        The velocity is computed as the dot product of the distribution functions and the lattice 
        velocities, divided by the density.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution functions.

        Returns
        -------
        rho: jax.numpy.ndarray
            Computed density.
        u: jax.numpy.ndarray
            Computed velocity.
        """
        rho = jnp.sum(f, axis=-1)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype).T
        u = jnp.dot(f, c) / rho[..., None]

        return rho, u
    
    @partial(jit, static_argnums=(0, 4), inline=True)
    def apply_bc(self, fout, fin, timestep, implementation_step):
        """
        This function applies the boundary conditions to the distribution functions.

        It iterates over all boundary conditions (BCs) and checks if the implementation step of the 
        boundary condition matches the provided implementation step. If it does, it applies the 
        boundary condition to the post-streaming distribution functions (fout).

        Parameters
        ----------
        fout: jax.numpy.ndarray
            The post-collision distribution functions.
        fin: jax.numpy.ndarray
            The post-streaming distribution functions.
        implementation_step: str
            The implementation step at which the boundary conditions should be applied.

        Returns
        -------
        ja.numpy.ndarray
            The output distribution functions after applying the boundary conditions.
        """
        for bc in self.BCs:
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementationStep == implementation_step:
                if bc.isDynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))
                    
        return fout

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, f_poststreaming, timestep, ret_fpost=False):
        """
        This function performs a single step of the LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions 
        towards their equilibrium values. It then applies the respective boundary conditions to the 
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution 
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming 
        distribution functions.

        Parameters
        ----------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions.
        timestep: int
            The current timestep of the simulation.
        ret_fpost: bool, optional
            If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if 
            ret_fpost is False.
        """
        f_postcollision = self.collision(f_poststreaming)
        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")

        if ret_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

    def run(self, t_max, print_iter=100, io_iter=100, io_ds_factor=1, MLUPS=False):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the 
        simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and 
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max: int
            The total number of time steps to run the simulation.
        print_iter: int, optional
            The number of time steps between each print of the simulation progress.
        io_iter: int, optional
            The number of time steps between each save of the simulation data.
        io_ds_factor: int, optional
            The downsampling factor for the simulation data.
        MLUPS: bool, optional
            If True, the function computes and prints the performance of the simulation in MLUPS.

        Returns
        -------
        f: jax.numpy.ndarray
            The distribution functions after the simulation.
        """
        f = self.assign_fields_sharding()
        u_prev, rho_prev = None, None

        if MLUPS:
            # Avoid io overhead
            io_iter = 0
            print_iter = 0
            start = time.time()

        # Loop over all time steps
        for timestep in range(t_max + 1):
            io_flag = io_iter > 0 and (timestep % io_iter == 0 or timestep == t_max)
            if io_flag:
                # Update the macroscopic variables and save the previous values
                rho, u = self.update_macroscopic(f)
                rho_prev, u_prev = np.array(process_allgather(rho)), np.array(process_allgather(u))
                rho_prev = downsample_scalarfield(rho_prev, io_ds_factor)
                u_prev = downsample_vectorfield(u_prev, io_ds_factor)
                del u, rho

            # Perform one time-step (collision, streaming, and boundary conditions)
            f, fstar = self.step(f, timestep, ret_fpost=self.ret_fpost)

            # Print the progress of the simulation
            if print_iter > 0 and timestep % print_iter == 0:
                print(f"{timestep}/{t_max} timesteps complete")

            if io_flag:
                # Save the simulation data
                print(f"Saving data at timestep {timestep}/{t_max}")
                rho, u = self.update_macroscopic(f)
                rho = downsample_scalarfield(rho, io_ds_factor)
                u = downsample_vectorfield(u, io_ds_factor)

                rho = np.array(process_allgather(rho))
                u = np.array(process_allgather(u))

                self.handle_io_timestep(timestep, f, fstar, rho, u, rho_prev, u_prev)
            
            if MLUPS and timestep == 1:
                jax.block_until_ready(f)
                start = time.time()

        if MLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f)
            end = time.time()
            if self.dim == 2:
                print('Domain: {} x {}'.format(self.nx, self.ny))
                print("Number of voxels: ", self.nx * self.ny)
                print(f"MLUPS: {self.nx * self.ny * t_max / (end - start) / 1e6}")
            elif self.dim == 3:
                print('Domain: {} x {} x {}'.format(self.nx, self.ny, self.nz))
                print("Number of voxels: ", self.nx * self.ny * self.nz)
                print(f"MLUPS: {self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}")
        return f

    def handle_io_timestep(self, timestep, f, fstar, rho, u, rho_prev, u_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten 
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.
        f: jax.numpy.ndarray
            The post-streaming distribution functions at the current time step.
        fstar: jax.numpy.ndarray
            The post-collision distribution functions at the current time step.
        rho: jax.numpy.ndarray
            The density field at the current time step.
        u: jax.numpy.ndarray
            The velocity field at the current time step.
        rho_prev: jax.numpy.ndarray
            The density field at the previous time step.
        u_prev: jax.numpy.ndarray
            The velocity field at the previous time step.
        """
        kwargs = {
            "timestep": timestep,
            "rho": rho,
            "u": u,
            "u_prev": u_prev,
            "rho_prev": rho_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar
        }
        self.output_data(**kwargs)

    def output_data(self, **kwargs):
        """
        This function is intended to be overwritten by the user to customize the input/output (I/O) 
        operations of the simulation.

        By default, it does nothing. When overwritten, it could save the simulation data to files, 
        display the simulation results in real time, send the data to another process for analysis, etc.

        Parameters
        ----------
        **kwargs: dict
            A dictionary containing the simulation data to be outputted. The keys are the names of the 
            data fields, and the values are the data fields themselves.
        """
        pass

    def set_boundary_conditions(self):
        """
        This function sets the boundary conditions for the simulation.

        It is intended to be overwritten by the user to specify the boundary conditions according to 
        the specific problem being solved.

        By default, it does nothing. When overwritten, it could set periodic boundaries, no-slip 
        boundaries, inflow/outflow boundaries, etc.
        """
        pass

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin):
        """
        This function performs the collision step in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the collision operator according to 
        the specific LBM model being used.

        By default, it does nothing. When overwritten, it could implement the BGK collision operator,
        the MRT collision operator, etc.

        Parameters
        ----------
        fin: jax.numpy.ndarray
            The pre-collision distribution functions.

        Returns
        -------
        fin: jax.numpy.ndarray
            The post-collision distribution functions.
        """
        pass

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, fpost_collision, feq, rho, u):
        """
        This function applies a force to the fluid in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the force according to the specific 
        problem being solved.

        By default, it does nothing and returns the input post-collision distribution functions 
        unchanged. When overwritten, it could implement a constant force, a time-dependent force, a 
        space-dependent force, etc.

        Parameters
        ----------
        fpost_collision: jax.numpy.ndarray
            The post-collision distribution functions.
        feq: jax.numpy.ndarray
            The equilibrium distribution functions.
        rho: jax.numpy.ndarray
            The density field.
        u: jax.numpy.ndarray
            The velocity field.

        Returns
        -------
        fpost_collision: jax.numpy.ndarray
            The post-collision distribution functions after the force is applied.
        """
        return fpost_collision