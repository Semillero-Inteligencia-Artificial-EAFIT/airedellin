import polars as pl
import glob

# List of file paths (assuming they are all CSV files)
file_paths = glob.glob("data/*.csv")

# Define the expected column names
columns = [
    "Fecha_Hora", "codigoSerial", "pm25", "calidad_pm25", "pm10", "calidad_pm10", "pm1", "calidad_pm1", 
    "no", "calidad_no", "no2", "calidad_no2", "nox", "calidad_nox", "ozono", "calidad_ozono", 
    "co", "calidad_co", "so2", "calidad_so2", "pst", "calidad_pst", "dviento_ssr", "calidad_dviento_ssr", 
    "haire10_ssr", "calidad_haire10_ssr", "p_ssr", "calidad_p_ssr", "pliquida_ssr", "calidad_pliquida_ssr", 
    "rglobal_ssr", "calidad_rglobal_ssr", "taire10_ssr", "calidad_taire10_ssr", "vviento_ssr", "calidad_vviento_ssr"
]
numeric_columns = [
    "pm25", "pm10", "pm1", "no", "no2", "nox", "ozono", "co", "so2", "pst", 
    "dviento_ssr", "haire10_ssr", "p_ssr", "pliquida_ssr", "rglobal_ssr", "taire10_ssr", "vviento_ssr"
]
dfs = []
for file_path in file_paths:
    try:
        # Attempt to read the file with headers
        df = pl.read_csv(file_path)
        
        # Check if the first column name matches the expected first column name
        if df.columns[0] != columns[0]:
            raise ValueError("Header mismatch")
    except:
        # Read the file without headers and assign the expected column names
        df = pl.read_csv(file_path, has_header=False)
        df.columns = columns
        
    for col in df.columns:
        df = df.with_columns(df[col].cast(pl.Utf8))

    dfs.append(df)
print(dfs)
# Concatenate all DataFrames into a single DataFrame
df = pl.concat(dfs)
#for i in numeric_columns:
#    df = df.with_columns(df[i].cast(pl.Float64))

print(df)
df.write_csv("all")
