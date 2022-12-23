import json
from typing import List, Dict, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from siml.siml_inferencer import WaterTankSimulator


class WaterTankSimulatorParameters(BaseModel):
    inlet_velocity: float


class SimulatorSettings(BaseModel):
    parameters: Union[WaterTankSimulatorParameters, None] = None
    eco_mode: bool = False


class SimulatorInput(BaseModel):
    parameters: Union[WaterTankSimulatorParameters, None] = None
    resolution: List[int] = [32, 32, 32]


app = FastAPI()


FAKE_SIMULATORS_DB = {
    "simulator123": WaterTankSimulator
}


LOADED_SIMULATORS = {}


@app.post("/init_simulator/{id}")
def simulate(id: str, settings: SimulatorSettings):
    if id not in FAKE_SIMULATORS_DB:
        raise HTTPException(status_code=404, detail="Simulator not found")

    simulator_loader = FAKE_SIMULATORS_DB.get(id)
    LOADED_SIMULATORS[id] = simulator_loader()
    LOADED_SIMULATORS[id].eco = settings.eco_mode
    LOADED_SIMULATORS[id].load_geometry()
    LOADED_SIMULATORS[id].load_inferencer()
    return {"message": "Simulator loaded."}


@app.post("/simulate/{id}")
def simulate(id: str, props: SimulatorInput):
    if id not in LOADED_SIMULATORS:
        raise HTTPException(status_code=404, detail="Simulator not loaded")

    simulator = LOADED_SIMULATORS[id]

    json_output = simulator.run_inference(props.parameters.inlet_velocity, props.resolution)
    
    return json_output


# kept for testing the endpoint
@app.get("/hello")
def read_root():
    return {"hello": "world"}
