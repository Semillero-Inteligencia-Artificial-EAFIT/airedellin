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

def statistics(data):
    if not data:
        return None
    total_sum=0
    max_value=0
    min_value=999999999999999999999999
    n = len(data)
    none_count=0
    for i in range(len(data)):
        total_sum+=data[i]
        if data[i]<min_value:
            min_value=data[i]
        if data[i]>max_value:
            max_value=data[i]
        if data[i]==None:
            none_count+=1

    mean = total_sum / n

    variance_sum=0
    # Initialize variables for higher moments
    m2, m3, m4 = 0, 0, 0  # second, third, and fourth moments
    for i in range(len(data)): 
        diff = data[i] - mean
        diff2 = diff ** 2
        diff3 = diff ** 3
        diff4 = diff ** 4
        m2 += diff2
        m3 += diff3
        m4 += diff4
    
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
        "mean": mean,
        "variance": variance,
        "standard_deviation": std_dev,
        "max": max_value,
        "min": min_value,
        "mode": mode,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "CV": cv,
        "z_scores": z_scores,
        "count_none": none_count
    }
