from typing import Union
import json
import numpy as np
from json import JSONEncoder
from fastapi import FastAPI

from siml.siml_inferencer import ModulusWaterTankRunner

app = FastAPI()


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def processData1D(data):
    lst = []
    for i in range(len(data['u'])):
        for index, item in enumerate(['u', 'v', 'w']):
            pts = data[item][i]
            lst.append(float(pts))
            if item == 'w':
                lst.append(float(1))
    
    return lst


@app.get("/hello")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/simulate")
def simulate():
    water_tank = ModulusWaterTankRunner()
    print(water_tank.eco)
    water_tank.load_geometry()
    water_tank.load_inferencer()
    output = water_tank.run_inference(5, [128, 128, 128])
    #print(output)
    output['u'] = np.reshape(output['u'], (-1, 1))
    output['v'] = np.reshape(output['v'], (-1, 1))
    output['w'] = np.reshape(output['w'], (-1, 1))
    output['v'] = np.reshape(output['v'], (-1, 1))
    #print(output['v'])
    #print(output['v'].shape)
    #print(processData1D(output))

    numpyData = {"array": [], "uvw": processData1D(output)}

    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)

    #return {"Hello": "World",}
    
    return encodedNumpyData
