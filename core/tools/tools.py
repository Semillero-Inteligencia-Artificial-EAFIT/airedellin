#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airellin - by jero98772
import datetime

def writetxt(name,content,mode="w"):
  """
  writetxt(name,content) , write in txt file something  
  """
  content=str(content)
  with open(name, mode) as file:
    file.write(content)
    file.close()



def nullValue(val,newval="-"):
    if not val or val=="":
        return newval 

def now():
  now = str(datetime.datetime.today().strftime("%m/%d/%Y, %H:%M"))
  return now

def readtxtline(name):
  with open(name, 'r') as file:
    return str(file.readline())

def divide_data(result_set):
    # Initialize empty lists for dates and pm2.5 values
    dates = []
    pm25_values = []
    
    # Extract data from the ResultSet
    for item in result_set["('fixed_stations_01', None)"]:
        print(item)
        # Append the time to the dates list
        dates.append(item['time'])
        # Append the pm2.5 value to the pm25_values list, handling None values
        pm25_values.append(item['data'] if item['data'] is not None else float('nan'))
    
    return dates, pm25_values