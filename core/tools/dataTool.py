from influxdb import InfluxDBClient
import random


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

    def coordinates(self, name):
        query = (
            "SELECT last(\"latitude\") AS \"lat\", last(\"longitude\") AS \"lon\" "
            f"FROM \"fixed_stations_01\" WHERE \"name\" = '{name}' AND time > now() - 1d"
        )
        result = self.client.query(query)
        points = list(result.get_points())
        if points:
            return points[0].get("lat"), points[0].get("lon")
        return None, None

    def station_data(self, name):
        coords = self.coordinates(name)
        pm25_data = self.data(name)
        return {
            "name": name,
            "coordinates": coords,
            "pm25_data": pm25_data
        }
    def get_formatted_data(self,size=50):
        features = []
        print(self.names())
        for name in self.names():
            coords = self.coordinates(name)
            pm25=self.data(name)[:10]
            filtered_arr = [x for x in pm25 if x is not None]
            #print(pm25)
            if not filtered_arr:
                mean_pm25 = 0  # Return None if there are no valid numbers
            else:
                mean_pm25 = sum(filtered_arr) / len(filtered_arr)
            print(mean_pm25)
            if len(self.data(name))==0:
                continue
            if coords[0] is not None and coords[1] is not None:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "name": name,
                        "pm25": float(mean_pm25)  # This will include all PM2.5 data points
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": coords
                    }
                }
                features.append(feature)
            else:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "name": name,
                        "pm25": float(mean_pm25)  # This will include all PM2.5 data points
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": generate_random_coordinates()
                    }
                }
                features.append(feature)
        return features