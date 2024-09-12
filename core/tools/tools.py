#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airellin - by MLeafit
import datetime
import numpy as np

def writetxt(name,content,mode="w"):
    """
    Write content to a text file.

    Parameters:
    name (str): The name of the file to write to.
    content (str or any): The content to be written. It will be converted to a string if necessary.
    mode (str, optional): The mode in which to open the file (default is "w" for writing).

    Returns:
    None
    """
    content=str(content)
    with open(name, mode) as file:
        file.write(content)
        file.close()

def nullValue(val,newval="-"):
    """
    Check if the value is null or empty and return a replacement value.

    Parameters:
    val (any): The value to check.
    newval (any, optional): The replacement value to return if val is null or empty (default is "-").

    Returns:
    any: newval if val is null or empty, otherwise returns val.
    """
    if not val or val=="":
        return newval 

def now():
    """
    Get the current date and time in a specific format.

    Returns:
    str: The current date and time formatted as 'MM/DD/YYYY, HH:MM'.
    """
    now = str(datetime.datetime.today().strftime("%m/%d/%Y, %H:%M"))
    return now

def readtxtline(name):
    """
    Read the first line from a text file.

    Parameters:
    name (str): The name of the file to read from.

    Returns:
    str: The first line of the file as a string.
    """
    with open(name, 'r') as file:
        return str(file.readline())

def divide_data(result_set):
    """
    Divide the result set into separate lists for dates and PM2.5 values.

    Parameters:
    result_set (dict): A dictionary containing the result set with keys 'time' and 'data'.

    Returns:
    tuple: A tuple containing two lists: dates and pm25_values.
    """
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
    """
    Extract statistics from a list of data points, including mean, variance, standard deviation, skewness, and kurtosis.
    i have to make this function because some tutorials of code use something very ineficient like:
    https://pastebin.com/HjgBFZKT this improve the performance x10 with this 
    
    Parameters:
    data (list): A list of numerical data points.

    Returns:
    dict: A dictionary containing statistical values such as mean, variance, standard deviation, kurtosis, skewness, etc.
    """
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
    """
    Determine the appropriate time range based on the input option.
    i did it here because for not fill post_sensor with Trash code 

    Parameters:
    range_option (str): A string representing the time range option ("1w", "1m", "1y").

    Returns:
    str: The corresponding time range in days, weeks, or hours.
    """
    if range_option == "1w":
        time_range = "7d"
    elif range_option == "1m":
        time_range = "4w"
    elif range_option == "1y":
        time_range = "182d"
    else:
        time_range = "24h"
    return time_range