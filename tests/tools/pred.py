from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

def timex2(time):
  newTime = []
  numVales = len(time)
  for i in time:
    newTime.append(numVales+i)
  return time + newTime

class pred():
	def __init__(self,data):
		self.data = data
	def predDataSarimax(self):
		self.timeser = list(range(len(self.data)))
		self.timeser2 = timex2(self.timeser) # time use for prediction
		#model = SARIMAX(endog = self.data,exog = self.timeser , order=(1,2,1), seasonal_order=(1,1,1,24))	
		model = SARIMAX(self.data, order=(0,0,0),seasonal_order=(1,1,1,24))
		self.predictions = list(model.fit().predict())
		return self.predictions
	"""
	def predDataLineal(self):
		import numpy as np
		timeser = np.asanyarray(self.timeser).reshape((1,-1))
		data = np.asanyarray(self.data).reshape((1,-1))
		timeser = timeser
		#y = data 
		sumy = np.sum(data)
		b = sumy/len(timeser)
		for i in range(len(timeser)):
		w = ((timeser[i]*timeser)-(pm25s[i]*pm25s))/((timeser[i]*timeser)**2)
		prediction = w*timeser+b
		for x in timeser:
			for y2 in y:
				filtred = np.polyfit(x, y2, 1)
		predictions = np.polyval(filtred, timeser)
		Y = np.polyval(predictions, timeser)
	"""
	def saveImg(self,pathName):	
		plt.xlabel("time (as amout of data)")
		plt.ylabel("pm25")
		plt.title("prediction")
		plt.plot(self.timeser,self.data,"bo",label="data")
		plt.plot(self.timeser2,self.data+self.predictions,"g-",label="prediction")
		plt.savefig(pathName+'.png')
		plt.clf() 
