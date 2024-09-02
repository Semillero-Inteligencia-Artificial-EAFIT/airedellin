import json
import os

# File to store the data
DATA_FILE = 'data_store.json'

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

def add_data(data):
    """Add new data to the data dictionary."""
    image = input("Enter image path: ")
    account = input("Enter account name: ")
    
    entity = input("Enter entity name: ")
    while entity in data['entities']:
        print("Entity name already exists. Please enter a unique name.")
        entity = input("Enter entity name: ")

    sensor = input("Enter sensor name: ")
    while sensor in data['sensors']:
        print("Sensor name already exists. Please enter a unique name.")
        sensor = input("Enter sensor name: ")

    data['images'].append(image)
    data['accounts'].append(account)
    data['entities'].append(entity)
    data['sensors'].append(sensor)
    
    save_data(data)

def edit_data(data):
    """Edit existing data."""
    for category in ['images', 'accounts', 'entities', 'sensors']:
        print(f"\nCurrent {category}:")
        for i, item in enumerate(data[category]):
            print(f"{i + 1}. {item}")

        edit_choice = input(f"Do you want to edit any {category}? (y/n): ").lower()
        if edit_choice == 'y':
            index = int(input(f"Enter the number of the {category[:-1]} you want to edit: ")) - 1
            new_value = input(f"Enter new value for {category[:-1]}: ")
            
            if category == "entities" or category == "sensors":
                while new_value in data[category]:
                    print(f"{category[:-1].capitalize()} name already exists. Please enter a unique name.")
                    new_value = input(f"Enter new value for {category[:-1]}: ")
                    
            data[category][index] = new_value

    save_data(data)

def main():
    """Main function to run the program."""
    data = load_data()
    
    if not data['images'] and not data['accounts'] and not data['entities'] and not data['sensors']:
        print("No data found. Let's start adding new data.")
        add_data(data)
    else:
        print("Data loaded successfully.")
        print("1. Add new data from scratch")
        print("2. Edit existing data")
        choice = input("Choose an option (1 or 2): ")
        
        if choice == '1':
            data = {"images": [], "accounts": [], "entities": [], "sensors": []}
            add_data(data)
        elif choice == '2':
            edit_data(data)
        else:
            print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()

