import json
import os
import random
import qrcode

# File to store the data
DATA_FILE = 'data_store.json'

# List of sensor names
SENSOR_NAMES = [
    'EZTTTGOTD0339E', 'EZTTTGOTD08C9E', 'EZTTTGOTDBD126', 'D29ESP32DE71402', 'D29ESP32DE7157E',
    'D29ESP32DE3B9CA', 'D29ESP32DE71DE2', 'D29ESP32DEFA43A', 'D29TTGOTD906BE', 'D29ESP32DEDF212',
    'D29ESP32DEFCA62', 'D29ESP32DED307A', 'D29ESP32DE73376', 'D29ESP32DE85636', 'EZTTTGOT770076',
    'D29ESP32DED1A1E', 'D29ESP32DE52BDE', 'D29ESP32DE3BD5E', 'C3NTTGOTD8E216', 'D29ESP32DE679CA',
    'D29ESP32DED2492', 'D29ESP32DED2FF6', 'D29ESP32DE3BCC6', 'D29ESP32DE0B7BE', 'D29ESP32DEC1106',
    'D29ESP32DED14D6', 'D29ESP32DE3BC5A', 'D29ESP32DED3782', 'D29ESP32DEFC17E', 'D29TTGOT7D532E',
    'D29ESP32DE5421A', 'D29ESP32DE53B7E', 'D29ESP32DEE1712', 'D29ESP32DE0C752', 'D29TTGOTD8F1AE',
    'D29ESP32DE3ADD6', 'D29ESP32DE3A682', 'EZTTTGOT7D5CFE', 'D29ESP32DE7298A', 'D29ESP32DE3BE12',
    'EZTESP32DE70042', 'D29ESP32DED2E9A', 'U0MM5STICK8DDB6', 'U33TTGOTDA585E', 'C3NTTGOTD1C61E',
    'D2GAGOPENC5F72', 'EZTTTGOTD71E6E', '6MCESP32DE1B00A'
]

def load_data():
    """Load data from the file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            return json.load(file)
    return {"images": [], "accounts": [], "entities": [], "sensors": []}

def save_data(data):
    """Save data to the file."""
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)
    print("Data saved successfully.")

def generate_qr_code(account,name):
    """Generate a QR code with the account information."""
    qr_data = f"{account}: don't work, this is a proof of concept"
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_data)
    qr.make(fit=True)
    
    img = qr.make_image(fill='black', back_color='white')
    img.save(name)
    print(f"QR code for {account} saved successfully.")

def add_random_data(data):
    """Add random data to the data dictionary."""
    for i in range(len(SENSOR_NAMES)):  # Adding 5 entries for demonstration
        sensor = SENSOR_NAMES[i]
        image = f"qr/image_{i}.png"
        account = f"account_{random.randint(1, 100)}"
        entity = f"entity_{random.randint(1, 100)}"
        
        data['images'].append(image)
        data['accounts'].append(account)
        data['entities'].append(entity)
        data['sensors'].append(sensor)
        
        generate_qr_code(account,image)

    save_data(data)

#def main():
    """Main function to run the program."""
"""
    data = load_data()
    
    if not data['images'] and not data['accounts'] and not data['entities'] and not data['sensors']:
        print("No data found. Adding random data.")
        add_random_data(data)
    else:
        print("Data loaded successfully.")
        print("1. Add new data from scratch")
        print("2. Edit existing data")
        choice = input("Choose an option (1 or 2): ")
        
        if choice == '1':
            data = {"images": [], "accounts": [], "entities": [], "sensors": []}
            add_random_data(data)
        elif choice == '2':
            edit_data(data)
        else:
            print("Invalid choice. Exiting.")
"""

def load_data():
    """Load data from the file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            return json.load(file)
    return {"images": [], "accounts": [], "entities": [], "sensors": []}

def retrieve_data_for_sensor(sensor_name):
    """Retrieve associated data based on the sensor name."""
    data = load_data()
    
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

def main():
    """Main function to run the program."""
    sensor_name = input("Enter the sensor name to retrieve its data: ")
    data = retrieve_data_for_sensor(sensor_name)
    
    if data:
        print("Retrieved Data:")
        for key, value in data.items():
            print(f"{key.capitalize()}: {value}")

if __name__ == '__main__':
    main()
