from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json

from .tools.dataTool import Sensors
# from .tools.pred import pred
# from .tools.const import *
from .tools.tools import *

app = FastAPI()

token = readtxtline("data/token.txt")
host = "influxdb.canair.io"
sensors = Sensors("canairio", host)
formatted_data = sensors.get_formatted_data()

templates = Jinja2Templates(directory="core/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    data = [
        {
            'type': 'Feature',
            'properties': {'name': 'Jardins du Trocadéro', 'pm25': 16},
            'geometry': {'type': 'Point', 'coordinates': [2.289207, 48.861561]},
        },
        {
            'type': 'Feature',
            'properties': {'name': 'Jardin des Plantes', 'pm25': 5},
            'geometry': {'type': 'Point', 'coordinates': [2.359823, 48.843995]},
        },
        {
            'type': 'Feature',
            'properties': {'name': 'Jardins das Tulherias', 'pm25': 9999},
            'geometry': {'type': 'Point', 'coordinates': [2.327092, 48.863608]},
        },
        {
            'type': 'Feature',
            'properties': {'name': 'Parc de Bercy', 'pm25': 12},
            'geometry': {'type': 'Point', 'coordinates': [2.382094, 48.835962]},
        },
        {
            'type': 'Feature',
            'properties': {'name': 'Jardin du Luxemburg', 'pm25': 6},
            'geometry': {'type': 'Point', 'coordinates': [2.336975, 48.846421]},
        },
    ] + formatted_data

    return templates.TemplateResponse("index.html", {"request": request, "token": token, "data": data})

@app.get("/{sensor_name}", response_class=HTMLResponse)
async def sensor(request: Request, sensor_name: str):
    return templates.TemplateResponse("sensors.html", {"request": request, "sensor_name": sensor_name})

@app.get("/{sensor_name}/{algorithm_name}", response_class=HTMLResponse)
async def mlalgorithm(request: Request, sensor_name: str, algorithm_name: str):
    return templates.TemplateResponse("algorithm.html", {"request": request, "algorithm_name": algorithm_name, "version": version})

