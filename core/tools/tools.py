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

