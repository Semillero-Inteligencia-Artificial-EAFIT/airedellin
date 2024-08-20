#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airdeellin - by MLEAFIT
from flask import Flask, render_template, request, flash, redirect ,session
import os
import pandas as pd
import polars as pl

from .tools.dataTool import sensors
#from .tools.pred import pred
from .tools.tools import *
from .tools.const import *

app = Flask(__name__)
class webpage():
  @app.route(WEBPAGE)
  def index():
    return render_template("index.html",token="")
