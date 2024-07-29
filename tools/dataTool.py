from influxdb import InfluxDBClient
class sensors:
	def __init__(self,db,host,port=8086):
		#self.port = port
		#self.host = host 
		self.client = InfluxDBClient(host=host, port=port,database=db)
		self.db = db
	def names(self):
		names = self.client.query('SHOW MEASUREMENTS ON "'+self.db+'"').raw
		clearNames = eval(str(names)[88:-4])
		return list(clearNames)
	def data(self,name,data = "pm25"):
		q = self.client.query('SELECT mean("'+data+'") AS "data" FROM "'+self.db+'"."autogen".'+name+' WHERE time > now() - 24h GROUP BY time(10s) FILL(none)')
		#q = self.client.query('select * from "PM2.5_BOG_FON_Hayuelos_E01" WHERE time >= now() - 10m')
		#q = self.client.query('SELECT mean("'+data+'") AS "data" FROM "'+self.db+'"."autogen".'+name+' WHERE (time > now() - 5h GROUP BY time(10s) FILL(none))')
		#q = self.client.query('select * from "'+name+'" WHERE time >= now() - 12h')
		#q = self.client.query('SELECT mean("'+data+'") AS "data" FROM "'+self.db+'"."autogen".'+name+' WHERE time > now() - 5h GROUP BY time(10s) FILL(none)')
		values = []
		#print(q)
		for value in q.get_points():
			values.append(value["data"])
		return values
"""
def test():
	#client = InfluxDBClient(host='influxdb.canair.io', port=8086, database='canairio')
	#host = "aqa.unloquer.org"
	# PM25_Berlin_CanAirIO_v2
	#dbs = sensors("canairio",host)
	dbs = sensors("canairio",host)
	print(dbs.data("PM25_Berlin_CanAirIO_v2"))
test()
"""