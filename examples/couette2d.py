from src.models import BGKSim
from src.boundary_conditions import *
from src.lattice import LatticeD2Q9
import jax.numpy as jnp
import numpy as np
from src.utils import *
from jax.config import config
import os

# config.update('jax_disable_jit', True)
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

precision = "f32/f32"


class Couette(BGKSim):

    def set_boundary_conditions(self):
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))

        self.BCs.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy))

        outlet = self.boundingBoxIndices["right"]
        inlet = self.boundingBoxIndices["left"]

        rho_wall = np.ones(inlet.shape[0], dtype=self.precision_policy.compute_dtype)
        vel_wall = np.zeros(inlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_wall[:, 0] = u_wall
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_wall, vel_wall))

        self.BCs.append(DoNothing(tuple(outlet.T), self.grid_info, self.precision_policy))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][..., 1:-1])
        u = np.array(kwargs["u"][..., 1:-1, :])
        timestep = kwargs["timestep"]
        u_prev = kwargs["u_prev"][..., 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)
        err = np.sum(np.abs(u_old - u_new))
        print("error= {:07.6f}".format(err))
        save_image(timestep, u)
        fields = {"rho": rho, "u_x": u[..., 0], "u_y": u[..., 1]}
        save_fields_vtk(timestep, fields)

if __name__ == "__main__":
    lattice = LatticeD2Q9(precision)

    nx = 501
    ny = 101

    Re = 100.0
    u_wall = 0.1
    clength = nx - 1

    visc = u_wall * clength / Re

    omega = 1.0 / (3.0 * visc + 0.5)
    print("omega = ", omega)
    assert omega < 1.98, "omega must be less than 2.0"
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")
    sim = Couette(lattice, omega, nx, ny, precision=precision)
    sim.run(20000, io_iter=10000)
