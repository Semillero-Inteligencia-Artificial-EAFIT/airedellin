#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airellin - by jero98772
import folium
import datetime

def writetxt(name,content,mode="w"):
  """
  writetxt(name,content) , write in txt file something  
  """
  content=str(content)
  with open(name, mode) as file:
    file.write(content)
    file.close()


def genMap(data,name):
  m = folium.Map(location=[6.256405968932449, -75.59835591123756])
  for i in range(len(data)):
    popup=data["name"][i]+"<li>"+data["contact"][i]+"</li><br>"
    folium.Marker([float(data["lng"][i]),float(data["lat"][i])], popup=popup, tooltip=data["name"][i]).add_to(m)
  m.save(name)


def nullValue(val,newval="-"):
    if not val or val=="":
        return newval 

def now():
  now = str(datetime.datetime.today().strftime("%m/%d/%Y, %H:%M"))
  return now

def readtxtline(name):
  with open(name, 'r') as file:
    return str(file.readline())

