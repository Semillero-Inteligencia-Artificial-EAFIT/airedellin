import json
import os


def load_data(data_file):
    """Load data from the file."""
    if os.path.exists(data_file):
        with open(data_file, 'r') as file:
            return json.load(file)
    return {"images": [], "accounts": [], "entities": [], "sensors": []}

def retrieve_data_for_sensor(sensor_name,data):
    """Retrieve associated data based on the sensor name."""
    #data = load_data("")
    
    if sensor_name not in data['sensors']:
        print(f"Sensor {sensor_name} not found.")
        return
    
    # Get index of the sensor
    index = data['sensors'].index(sensor_name)
    
    # Retrieve associated data
    associated_data = {
        "sensor": sensor_name,
        "entity": data['entities'][index] if index < len(data['entities']) else None,
        "account": data['accounts'][index] if index < len(data['accounts']) else None,
        "image": data['images'][index] if index < len(data['images']) else None
    }
    
    return associated_data
