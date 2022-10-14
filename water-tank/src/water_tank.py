from sympy import Symbol
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.key import Key
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.basic import NormalDotVec

# params
# Water at 20°C (https://wiki.anton-paar.com/en/water/)
# https://en.wikipedia.org/wiki/Viscosity#Kinematic_viscosity
nu = 1.787e-06  # m2 * s-1
inlet_vel = Symbol("inlet_velocity")
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

# parameterization
inlet_vel_range = (0.05, 10.0)
inlet_vel_params = {inlet_vel: inlet_vel_range}


def network(cfg: ModulusConfig, scale):
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu * scale, rho=1.0, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("inlet_velocity")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    return (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network", jit=cfg.jit)]
    )


def constraints(cfg: ModulusConfig, geo, nodes, domain):
    # add constraints to solver
    # inlet
    u, v, w = geo.circular_parabola(
        x,
        y,
        z,
        center=geo.inlet_center,
        normal=geo.inlet_normal,
        radius=geo.inlet_radius,
        max_vel=inlet_vel,
    )

    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo.inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.inlet,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo.outlet_left_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(outlet_left, "outlet_left")

    outlet_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo.outlet_right_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(outlet_right, "outlet_right")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo.noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=inlet_vel_params
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo.interior_mesh,
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
        geometry=geo.outlet_left_mesh,
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
        geometry=geo.outlet_right_mesh,
        outvar={"normal_dot_vel": 2.540},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        parameterization=inlet_vel_params
    )
    domain.add_constraint(integral_continuity_outlet_right, "integral_continuity_3")
