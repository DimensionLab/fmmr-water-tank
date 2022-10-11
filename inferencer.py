from modulus.hydra import to_yaml
from modulus.hydra.utils import compose
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.inferencer import PointwiseInferencer

from src.geometry import WaterTank
from src.water_tank import network, constraints, inlet_vel
from src.plotter import generate_velocity_profile_3d, InferencerSlicePlotter2D


cfg = compose(config_path="conf", config_name="config_eval", job_name="water_tank_inference")
print(to_yaml(cfg))


def run():
    geo = WaterTank()
    domain = Domain()
    nodes = network(cfg, scale=geo.scale)
    constraints(cfg, geo=geo, nodes=nodes, domain=domain)

    inlet_vel_inference = 2.0

    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=geo.interior_mesh.sample_interior(1000, parameterization={inlet_vel: inlet_vel_inference}),
        output_names=["u", "v", "w", "p"],
        batch_size=1024,
        plotter=InferencerSlicePlotter2D()
    )
    domain.add_inferencer(inferencer, "simulation")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

    # generate velocity profile with magnitude (it has V = [u, v, w] in one array)
    generate_velocity_profile_3d()


if __name__ == "__main__":
    run()
