#!/usr/bin/env python
# -*- coding: utf-8 -*-"
#airellin - by MLeafit
import json
import os
def load_data(data_file):
    """
    Load data from a JSON file if it exists.
    
    Args:
        data_file (str): The path to the JSON file that contains data.

    Returns:
        dict: A dictionary containing keys 'images', 'accounts', 'entities', and 'sensors'.
        If the file does not exist, it returns an empty structure with those keys and empty lists.
    """
    if os.path.exists(data_file):
        with open(data_file, 'r') as file:
            return json.load(file)
    return {"images": [], "accounts": [], "entities": [], "sensors": []}

def retrieve_data_for_sensor(sensor_name,data):
    """
    Retrieve data associated with a sensor by its name.

    Args:
        sensor_name (str): The name of the sensor to look up in the data.
        data (dict): A dictionary containing 'sensors', 'entities', 'accounts', and 'images' lists.

    Returns:
        dict: A dictionary containing the sensor name, its associated entity, account, and image.
        If the sensor is not found, returns a placeholder dictionary with a message for donation.

    Example:
        If the sensor exists:
        {
            "sensor": "sensor_name",
            "entity": "corresponding_entity",
            "account": "corresponding_account",
            "image": "corresponding_image_url"
        }

        If the sensor does not exist:
        {
            "sensor": "not found",
            "entity": "you can add your sensor for donations",
            "account": "go to <a href='/add_donation'>add donation</a>",
            "image": "https://via.placeholder.com/150"
        }
    """
    if sensor_name not in data['sensors']:
        print(f"Sensor {sensor_name} not found.")
        associated_data = {
            "sensor": "not found",
            "entity": "you can add your sensor for donatios",
            "account": "go to <a href='/add_donation'>add donation</a>",
            "image": "https://via.placeholder.com/150",
        }
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
