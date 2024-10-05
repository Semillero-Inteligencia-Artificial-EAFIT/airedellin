import h3
import polars as pl

def get_pm25_features(file_path: str, resolution: int = 3):
    """
    Reads a CSV file containing latitude, longitude, and PM2.5 data, groups the data by H3 hexagon IDs,
    and returns a dictionary with hexagon IDs as keys and lists of PM2.5 values as values.

    Parameters:
    ----------
    file_path : str
        The path to the CSV file containing columns 'lat', 'lon', and 'MERRA2_CNN_Surface_PM25'.
    resolution : int, optional
        The resolution level for the H3 hexagons, by default 3. Higher values create smaller hexagons.

    Returns:
    --------
    Dict[str, List[float]]:
        A dictionary where keys are H3 hexagon IDs and values are lists of PM2.5 values for that hexagon.
    """
    # Read the CSV file with Polars
    df = pl.read_csv(file_path)
    #print(df)

    # Initialize an empty dictionary to store features
    features = {}

    # Iterate through each row in the dataframe
    for row in df.iter_rows():
        lat, lon, pm25 = row[df.columns.index("lat")], row[df.columns.index("lon")], row[df.columns.index("MERRA2_CNN_Surface_PM25")]
        
        # Generate H3 hexagon ID for the given coordinates
        hexId = h3.geo_to_h3(lat, lon, resolution)
        
        # Add the PM2.5 value to the list for this hexagon ID
        if hexId in features:
            features[hexId].append(pm25)
        else:
            features[hexId] = [pm25]

    return features

# Example usage
my_dict=get_pm25_features("data_nasa.csv")
keys = list(my_dict.keys())
values = list(my_dict.values())
print(my_dict)
print(keys)
print(values)
#print(get_pm25_features("data_nasa.csv"))