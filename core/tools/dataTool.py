#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airellin - by MLeafit

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
    """
    A class to interact with an InfluxDB instance and retrieve sensor data from fixed stations.

    Attributes:
    - client (InfluxDBClient): The client object for interacting with the InfluxDB.
    - db (str): The database name.
    - table (str): The table to query data from.
    """
    def __init__(self, db, host, port=8086):
        """
        Initializes the Sensors object with a database, host, and port.

        Parameters:
        - db (str): The name of the database to connect to.
        - host (str): The hostname of the InfluxDB server.
        - port (int, optional): The port number to connect to. Default is 8086.
        """
        self.client = InfluxDBClient(host=host, port=port, database=db)
        self.db = db
        self.table="fixed_stations_01"

    def names(self):
        """
        Retrieves a list of distinct station names from the database within the last 7 days(because i wanna plot active sensors).

        Returns:
        - list: A list of distinct station names.
        """
        query = 'SELECT DISTINCT("name") FROM "fixed_stations_01" WHERE time > now() - 7d'
        result = self.client.query(query)
        return [item['distinct'] for item in result.get_points()]

    async def data(self, name,time="24h",date=False):
        """
        Retrieves PM2.5 data for a specific station within the specified time range.

        Parameters:
        - name (str): The name of the station to query.
        - time (str, optional): The time range to query, default is 24 hours.
        - date (bool, optional): Whether to return timestamps along with the data. Default is False.

        Returns:
        - list: A list of PM2.5 values.
        - tuple (list, list): If date=True, returns two lists: one for PM2.5 values and another for timestamps.
        """
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
            return [value["data"] for value in result.get_points()]
    def data(self, name,time="24h",date=False):
        """
        Retrieves PM2.5 data for a specific station within the specified time range.

        Parameters:
        - name (str): The name of the station to query.
        - time (str, optional): The time range to query, default is 24 hours.
        - date (bool, optional): Whether to return timestamps along with the data. Default is False.

        Returns:
        - list: A list of PM2.5 values.
        - tuple (list, list): If date=True, returns two lists: one for PM2.5 values and another for timestamps.
        """
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
            return [value["data"] for value in result.get_points()]

    async def data_complate_particules(self, name,time="24h",date=False):
        """
        Retrieves PM2.5, PM10, and PM1 data for a specific station within the specified time range.

        Parameters:
        - name (str): The name of the station to query.
        - time (str, optional): The time range to query, default is 24 hours.
        - date (bool, optional): Whether to return timestamps along with the data. Default is False.

        Returns:
        - dict: A dictionary containing lists of PM2.5, PM10, and PM1 values.
        - dict: If date=True, returns a dictionary with dates and PM2.5, PM10, PM1 values.
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
                if value["time"]!=None:
                    date.append(value["time"])  
            return {"dates":date,"pm25":pm25,"pm10":pm10,"pm1":pm1}
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
        """
        Retrieves the geohash coordinates of a station from the database.

        Parameters:
        - name (str): The name of the station to query.

        Returns:
        - tuple: A tuple containing the longitude and latitude coordinates of the station.
        - (None, None): If no coordinates are found.
        """
        query = (
            "SELECT last(\"geo\") AS \"geohash\""
            f" FROM \"fixed_stations_01\" WHERE \"name\" = '{name}' AND time > now() - 1d"
        )
        result = self.client.query(query)
        points = list(result.get_points())
        if points:
            cords=pgh.decode(points[0].get("geohash"))
            return cords[1],cords[0]
        return None,None


    async def station_data(self, name):
        """
        Retrieves the station's name, coordinates, and PM2.5 data.

        Parameters:
        - name (str): The name of the station to retrieve data for.

        Returns:
        - dict: A dictionary containing the station name, coordinates, and PM2.5 data.
        """
        coords = self.coordinates(name)
        pm25_data = self.data(name)
        return {
            "name": name,
            "coordinates": coords,
            "pm25_data": pm25_data
        }

    async def get_formatted_data(self, size=50):
        """
        Asynchronously retrieves formatted data from stations, including name, PM2.5 data, and coordinates.

        Parameters:
        - size (int, optional): Number of stations to process. Default is 50.

        Returns:
        - list: A list of GeoJSON feature dictionaries containing the station name, PM2.5 value, and coordinates.
        """
        features = []
        
        for name in self.names():  # Iterate over sensor names
            coords = self.coordinates(name)
            pm25 = self.data(name)[:10]  
            filtered_arr = [x for x in pm25 if x is not None]  # Filter out None values

            if not filtered_arr:
                mean_pm25 = 0  # Set mean to 0 if there are no valid values
            else:
                mean_pm25 = sum(filtered_arr) / len(filtered_arr)  # Calculate the mean
            
            
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
