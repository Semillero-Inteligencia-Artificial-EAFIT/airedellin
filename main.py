#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airdeellin - by MLEAFIT
from flask import Flask, render_template, request, flash, redirect ,session
import os
import pandas as pd
import polars as pl

from tools.dataTool import sensors
from tools.pred import pred
from tools.tools import *
from tools.const import *

app = Flask(__name__)
class webpage():

    if not os.path.isfile(DATAPATH):
      writetxt(DATAPATH,COLSDATA)
    class webpage():
      @app.route(WEBPAGE)
      def index():
        return render_template("index.html")

      @app.route(WEBPAGE+"/map")
      def mapweb():
        return render_template(MAPNAME)

      @app.route("/addData.html",methods=["GET","POST"])
      def addData():
        if request.method=="POST":
         name=request.form["name"]
         website=request.form["website"]
         prediction=request.form["prediction"]
         lat=request.form["lat"]
         lng=request.form["lng"]
         data=name+","+website+","+prediction+","+lat+","+lng+"\n"
         writetxt(DATAPATH,data,"a")
         df=pd.read_csv(DATAPATH)
         print()
         genMap(df,"templates/"+MAPNAME)
        return render_template("addData.html")

    @app.route(WEBPAGE+"prediction.html")
    def prediction():
        status = ""
        sensorsNames = []
        sensorsON = []
        for i in dbNames:
            db = sensors(i,HOST)
            sensorsNames += db.names()
            print(sensorsNames)
            for ii in sensorsNames:
                #print(ii)
                name = ii[0]
                pm25 = db.data(name)
                print(pm25)

        return render_template("consultor.html",names = sensorsON,msg = status)
webpage()
      
if __name__ == "__main__":
  app.run(debug=True,host="127.0.0.1",port=5000)