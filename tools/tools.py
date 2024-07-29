#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#commap - by jero98772
def writetxt(name,content,mode="w"):
  """
  writetxt(name,content) , write in txt file something  
  """
  content=str(content)
  with open(name, mode) as file:
    file.write(content)
    file.close()
def genMap(data,name):
  import folium
  m = folium.Map(location=[6.256405968932449, -75.59835591123756])
  #folium.TileLayer('Mapbox Control Room').add_to(m)
  for i in range(len(data)):
    popup=data["name"][i]+"<li>"+data["contact"][i]+"</li><br>"
    folium.Marker([float(data["lng"][i]),float(data["lat"][i])], popup=popup, tooltip=data["name"][i]).add_to(m)
  m.save(name)
def nullValue(val,newval="-"):
    if not val or val=="":
        return newval 