import numpy as np
from sympy import sqrt, Max
from modulus.hydra import to_absolute_path
from modulus.geometry.tessellation import Tessellation
from modulus.utils.io.vtk import var_to_polyvtk


class WaterTank:
    """Water tank geometry"""
    inlet_area = None

    def __init__(self):
        # read stl files to make geometry
        point_path = to_absolute_path("./stl_files")

        inlet_mesh = Tessellation.from_stl(
            point_path + "/inlet.stl", airtight=False
        )
        outlet_left_mesh = Tessellation.from_stl(
            point_path + "/outlet_left.stl", airtight=False
        )
        outlet_right_mesh = Tessellation.from_stl(
            point_path + "/outlet_right.stl", airtight=False
        )
        noslip_mesh = Tessellation.from_stl(
            point_path + "/water_tank_noslip.stl", airtight=False
        )
        interior_mesh = Tessellation.from_stl(
            point_path + "/water_tank_closed.stl", airtight=True
        )

        # scale and normalize mesh and openfoam data
        self.center = (0, 0, 0)
        self.scale = 1.0
        self.inlet_mesh = self.normalize_mesh(inlet_mesh, self.center, self.scale)
        self.outlet_left_mesh = self.normalize_mesh(outlet_left_mesh, self.center, self.scale)
        self.outlet_right_mesh = self.normalize_mesh(outlet_right_mesh, self.center, self.scale)
        self.noslip_mesh = self.normalize_mesh(noslip_mesh, self.center, self.scale)
        self.interior_mesh = self.normalize_mesh(interior_mesh, self.center, self.scale)

        # geom params
        self.inlet_normal = (0.0, 0.0, -2.0)
        self.inlet_center = (0.0, 0.0, 3.0)
        s = inlet_mesh.sample_boundary(nr_points=10000)
        self.inlet_area = np.sum(s["area"])
        print("Surface Area: {:.3f}".format(self.inlet_area))
        self.inlet_radius = np.sqrt(self.inlet_area / np.pi)

        s = self.interior_mesh.sample_interior(nr_points=10000, compute_sdf_derivatives=True)
        print("Volume: {:.3f}".format(np.sum(s["area"])))

    # inlet velocity profile
    def circular_parabola(self, x, y, z, center, normal, radius, max_vel):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]
        distance = sqrt(centered_x ** 2 + centered_y ** 2 + centered_z ** 2)
        parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    # normalize meshes
    def normalize_mesh(self, mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

    # normalize invars
    def normalize_invar(self, invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale ** dims
        return invar
