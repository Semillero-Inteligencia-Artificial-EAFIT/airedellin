#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airellin - by MLeafit
import datetime
import numpy as np

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

def statistics_extractor(data):
    new_data=[]
    if not data:
        return None
    total_sum=0
    max_value=0
    min_value=9999999999999999999999999999
    n = len(data)
    none_count=0
    ocurrences={}
    for i in range(len(data)):
        if data[i]==None:
            none_count+=1
        else:

            if data[i] in ocurrences:
                ocurrences[int(data[i])]+=1
            else:
                ocurrences[int(data[i])]=1
            total_sum+=data[i]
            if data[i]<min_value:
                min_value=data[i]
            if data[i]>max_value:
                max_value=data[i]
            new_data.append(data[i])
    mean = total_sum / n

    data=new_data
    variance_sum=0
    # Initialize variables for higher moments
    m2, m3, m4 = 0, 0, 0  # second, third, and fourth moments
    std_deviations=[]
    for i in range(len(data)): 
        diff = data[i] - mean
        m2 += diff ** 2
        m3 += diff ** 3
        m4 += diff ** 4
        #z_scores.append(diff/std_dev)
        std_deviations.append(diff)
    
    # Variance, standard deviation, skewness, and kurtosis
    variance = m2 / n
    std_dev = variance ** 0.5
    skewness = (m3 / n) / (std_dev ** 3)
    kurtosis = (m4 / n) / (variance ** 2) - 3

    
    # Mode
    mode = max(set(data), key=data.count)
    cv = std_dev / mean
    z_scores = [(x - mean) / std_dev for x in data]

    return {
        "sum": total_sum,
        "mean":round(mean,4),
        "variance": round(variance,4),
        "standard_deviation": round(std_dev,4),
        "max": max_value,
        "min": min_value,
        "mode": mode,
        "kurtosis": round(kurtosis,4),
        "skewness": round(skewness,4),
        "CV": round(cv,4),
        "count_none": none_count,
        "z_scores": z_scores,
        "ocurrences": ocurrences,

    }

def range_option_function(range_option):
    if range_option == "1w":
        time_range = "7d"
    elif range_option == "1m":
        time_range = "4w"
    elif range_option == "1y":
        time_range = "182d"
    else:
        time_range = "24h"
    return time_range