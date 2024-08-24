from influxdb import InfluxDBClient

class Sensors:
    def __init__(self, db, host, port=8086):
        self.client = InfluxDBClient(host=host, port=port, database=db)
        self.db = db

    def names(self):
        query = 'SELECT DISTINCT("name") FROM "fixed_stations_01" WHERE time > now() - 1d'
        result = self.client.query(query)
        return [item['distinct'] for item in result.get_points()]

    def data(self, name):
        query = (
            "SELECT mean(\"pm25\") AS \"data\" FROM \"fixed_stations_01\" "
            f"WHERE \"name\" = '{name}' AND time >= now() - 24h "
            "GROUP BY time(30s) fill(null) ORDER BY time ASC"
        )
        result = self.client.query(query)
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
