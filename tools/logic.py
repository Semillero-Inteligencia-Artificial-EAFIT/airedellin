def now():
	import datetime
	now = str(datetime.datetime.today().strftime("%m/%d/%Y, %H:%M"))
	return now
def readtxtline(name):
	with open(name, 'r') as file:
		return str(file.readline())
def writetxt(name,content):
	content = str(content)
	with open(name, 'w') as file:
		file.write(content)
		file.close()
def timex2(time):
	newTime = []
	numVales = len(time)
	for i in time:
		newTime.append(numVales+i)
	return time + newTime
