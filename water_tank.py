import torch
import numpy as np
from sympy import Symbol, sqrt, Max

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.domain.monitor import PointwiseMonitor
from modulus.domain.inferencer import PointwiseInferencer
from modulus.domain.inferencer import PointVTKInferencer
from modulus.utils.io import (
    VTKUniformGrid,
)
from modulus.utils.io.vtk import var_to_polyvtk, VTKFromFile
from modulus.key import Key
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.basic import NormalDotVec
from modulus.utils.io import csv_to_dict
from modulus.geometry.tessellation import Tessellation


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

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

    # sample geometry for plotting in Paraview
    s = inlet_mesh.sample_boundary(nr_points=10000)
    var_to_polyvtk(s, "inlet_boundary")
    inlet_area = np.sum(s["area"])
    print("Surface Area: {:.3f}".format(inlet_area))
    s = interior_mesh.sample_interior(nr_points=10000, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "interior")
    print("Volume: {:.3f}".format(np.sum(s["area"])))

    # params
    # Water at 20°C (https://wiki.anton-paar.com/en/water/)
    # https://en.wikipedia.org/wiki/Viscosity#Kinematic_viscosity
    nu = 1.787e-06  # m2 * s-1
    inlet_vel = Symbol("inlet_velocity")
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    # parameterization
    inlet_vel_range = (0.05, 10.0)
    inlet_vel_params = {inlet_vel: inlet_vel_range}

    # inlet velocity profile
    def circular_parabola(x, y, z, center, normal, radius, max_vel):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]
        distance = sqrt(centered_x ** 2 + centered_y ** 2 + centered_z ** 2)
        parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

    # normalize invars
    def normalize_invar(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale ** dims
        return invar

    # scale and normalize mesh and openfoam data
    center = (0, 0, 0)
    scale = 1.0
    inlet_mesh = normalize_mesh(inlet_mesh, center, scale)
    outlet_left_mesh = normalize_mesh(outlet_left_mesh, center, scale)
    outlet_right_mesh = normalize_mesh(outlet_right_mesh, center, scale)
    noslip_mesh = normalize_mesh(noslip_mesh, center, scale)
    interior_mesh = normalize_mesh(interior_mesh, center, scale)

    # geom params
    inlet_normal = (0.0, 0.0, -2.0)
    inlet_center = (0.0, 0.0, 3.0)
    inlet_radius = np.sqrt(inlet_area / np.pi)

    # make aneurysm domain
    domain = Domain()

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu * scale, rho=1.0, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("inlet_velocity")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network", jit=cfg.jit)]
    )

    # add constraints to solver
    # inlet
    u, v, w = circular_parabola(
        x,
        y,
        z,
        center=inlet_center,
        normal=inlet_normal,
        radius=inlet_radius,
        max_vel=inlet_vel,
    )
    print(u)
    print(v)
    print(w)
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.inlet,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_left_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(outlet_left, "outlet_left")

    outlet_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_right_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(outlet_right, "outlet_right")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(interior, "interior")

    # Integral Continuity 1
    # TODO: add integral plane somewhere into the geometry

    # Integral Continuity 2
    integral_continuity_outlet_left = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_left_mesh,
        outvar={"normal_dot_vel": 2.540},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        parameterization=inlet_vel_params
    )
    domain.add_constraint(integral_continuity_outlet_left, "integral_continuity_2")

    # Integral Continuity 3
    integral_continuity_outlet_right = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_right_mesh,
        outvar={"normal_dot_vel": 2.540},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        parameterization=inlet_vel_params
    )
    domain.add_constraint(integral_continuity_outlet_right, "integral_continuity_3")

    inlet_vel_inference = 1.5

    # add inferencer
    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=interior_mesh.sample_interior(1000000, parameterization={inlet_vel: inlet_vel_inference}),
        output_names=["u", "v", "w", "p"],
        batch_size=4096,
    )
    domain.add_inferencer(inferencer, "inf_data")

    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=interior_mesh.sample_interior(5000, parameterization={inlet_vel: inlet_vel_inference}),
        output_names=["u", "v", "w", "p"],
        batch_size=256,
    )
    domain.add_inferencer(inferencer, "inf_data_small")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
