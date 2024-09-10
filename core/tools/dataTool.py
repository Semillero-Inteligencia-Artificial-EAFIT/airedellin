from influxdb import InfluxDBClient
import random
import pygeohash as pgh


def generate_random_coordinates(x_range=(-80, 80), y_range=(-60, 60)):
    """
    Generates a tuple representing random coordinates within the given x and y ranges.

    Parameters:
    - x_range (tuple): A tuple specifying the min and max range for x coordinates.
    - y_range (tuple): A tuple specifying the min and max range for y coordinates.

    Returns:
    - tuple: A tuple containing the x and y coordinates.
    """
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    return (x, y)


class Sensors:
    def __init__(self, db, host, port=8086):
        self.client = InfluxDBClient(host=host, port=port, database=db)
        self.db = db
        self.table="fixed_stations_01"

    def names(self):
        print(self.db)
        query = 'SELECT DISTINCT("name") FROM "fixed_stations_01" WHERE time > now() - 7d'
        result = self.client.query(query)
        return [item['distinct'] for item in result.get_points()]
#    def names(self):
        """
        Retrieves all measurement names from the specified InfluxDB database.

        :return: List of measurement names as strings.
        """
        """
        query = f'SHOW MEASUREMENTS ON "{self.db}"'
        result = self.client.query(query)
        names = [item['name'] for item in result.get_points()]
        return names
        """
    def data(self, name,time="24h",date=False):
        query = (
            "SELECT mean(\"pm25\") AS \"data\" FROM \"fixed_stations_01\" "
            f"WHERE \"name\" = '{name}' AND time >= now() - {time} "
            "GROUP BY time(30s) fill(null) ORDER BY time ASC"
        )
        if date:
            result = self.client.query(query)
            pm25=[]
            date=[]
            for value in result.get_points():
                date.append(value["time"])
                pm25.append(value["data"])
            return pm25,date
        else:
            result = self.client.query(query)
            #print(list(result.get_points()))
            return [value["data"] for value in result.get_points()]
    def data_complate_particules(self, name,time="24h",date=False):
        """
        To test
        """
        query = (
            "SELECT mean(\"pm25\") AS \"mean_pm25\", "
            "mean(\"pm10\") AS \"mean_pm10\", "
            "mean(\"pm1\") AS \"mean_pm1\" "
            "FROM \"fixed_stations_01\" "
            f"WHERE \"name\" = '{name}' AND time >= now() - {time} "
            "GROUP BY time(30s) fill(null) "
            "ORDER BY time ASC"
        )
        if date:
            result = self.client.query(query)
            pm25=[]
            pm10=[]
            pm1=[]
            date=[]
            for value in result.get_points():
                if value["mean_pm25"]!=None:
                    pm25.append(value["mean_pm25"])
                if value["mean_pm10"]!=None:
                    pm10.append(value["mean_pm10"])
                if value["mean_pm1"]!=None:
                    pm1.append(value["mean_pm1"])  

            return {"date":date,"pm25":pm25,"pm10":pm10,"pm1":pm1}
        else:
            result = self.client.query(query)
            pm25=[]
            pm10=[]
            pm1=[]
            date=[]
            for value in result.get_points():
                if value["mean_pm25"]!=None:
                    pm25.append(value["mean_pm25"])
                if value["mean_pm10"]!=None:
                    pm10.append(value["mean_pm10"])
                if value["mean_pm1"]!=None:
                    pm1.append(value["mean_pm1"])            
            return {"pm25":pm25,"pm10":pm10,"pm1":pm1}

    def coordinates(self, name):
        query = (
            "SELECT last(\"geo\") AS \"geohash\""
            f" FROM \"fixed_stations_01\" WHERE \"name\" = '{name}' AND time > now() - 1d"
        )
        result = self.client.query(query)
        #print(result)  # Print the raw result for debugging
        points = list(result.get_points())
        #print(points)
        if points:
            cords=pgh.decode(points[0].get("geohash"))
            print(cords)
            return cords[1],cords[0]
        return None,None


    def station_data(self, name):
        coords = self.coordinates(name)
        pm25_data = self.data(name)
        return {
            "name": name,
            "coordinates": coords,
            "pm25_data": pm25_data
        }
    async def get_formatted_data(self, size=50):
        features = []
        print(self.names())  # Assuming self.names() is async, else remove await
        
        for name in self.names():  # Iterate over sensor names
            coords = self.coordinates(name)
            print(coords)  
            pm25 = self.data(name)[:10]  
            filtered_arr = [x for x in pm25 if x is not None]  # Filter out None values

            if not filtered_arr:
                mean_pm25 = 0  # Set mean to 0 if there are no valid values
            else:
                mean_pm25 = sum(filtered_arr) / len(filtered_arr)  # Calculate the mean
            
            print(mean_pm25)
            
            if len(pm25) == 0:  # Skip if no PM2.5 data
                continue
            
            if coords[0] is not None and coords[1] is not None:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "name": name,
                        "pm25": float(mean_pm25)  # PM2.5 average
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": coords  # Valid coordinates
                    }
                }
            else:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "name": name,
                        "pm25": float(mean_pm25)  # PM2.5 average
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": generate_random_coordinates()  # Generate random coordinates if missing
                    }
                }
            
            features.append(feature)
        
        return features
