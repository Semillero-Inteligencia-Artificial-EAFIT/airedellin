#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airdeellin - by MLEAFIT
from flask import Flask, render_template, request, flash, redirect ,session
import json


from .tools.dataTool import Sensors
#from .tools.pred import pred
#from .tools.const import *
from .tools.tools import *



app = Flask(__name__)
token=readtxtline("data/token.txt")
host = "influxdb.canair.io"
sensors = Sensors("canairio", host)
formatted_data = sensors.get_formatted_data()
        
class webpage():
  WEBPAGE="/"
  @app.route(WEBPAGE)
  def index():
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
  ]+formatted_data

          
    
    return render_template("index.html",token=token,data=data)
    
  @app.route(WEBPAGE + "<string:sensor_name>")
  def sensor(sensor_name):
      return render_template("sensors.html", sensor_name=sensor_name)

  @app.route(WEBPAGE + "<string:sensor_name>/<string:algorithm_name>")
  def mlalgorithm(sensor_name, algorithm_name):
      return render_template("algorithm.html", algorithm_name=algorithm_name, version=version)
