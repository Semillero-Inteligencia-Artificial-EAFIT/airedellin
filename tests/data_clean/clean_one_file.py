import polars as pl

# Define the required columns
required_columns = [
    "Fecha_Hora", "codigoSerial", "pm25", "calidad_pm25", "pm10", "calidad_pm10", "pm1", "calidad_pm1"
]

# Read CSV with schema overrides
df = pl.read_csv(
    "cleaned_filtered_data.csv",
    schema_overrides={
        "Fecha_Hora": pl.Utf8,
        "codigoSerial": pl.Utf8,
        "pm25": pl.Utf8,
        "calidad_pm25": pl.Utf8,
        "pm10": pl.Utf8,
        "calidad_pm10": pl.Utf8,
        "pm1": pl.Utf8,
        "calidad_pm1": pl.Utf8,
    },
    ignore_errors=True
)

def clean_and_cast_column(df, col):
    # Remove unwanted values and cast to float
    df = df.filter(pl.col(col) != col)
    df = df.with_columns(
        pl.col(col).str.replace(".", "", literal=True).str.replace(",", ".", literal=True).cast(pl.Float64).alias(col)
    )
    
    # Replace values based on conditions
    df = df.with_columns(
        pl.when(
            (pl.col(col) > 1999) | 
            (pl.col(col).is_in([-1, -99990])) | 
            (pl.col(col) < 0) | 
            (pl.col(col).is_in([200, 9999999]))
        )
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
    )
    
    # Impute missing values with column mean
    col_mean = df[col].mean()
    df = df.with_columns(
        pl.when(pl.col(col).is_null())
        .then(col_mean)
        .otherwise(pl.col(col))
        .alias(col)
    )
    
    return df

# Clean and cast numeric columns
for col in ["pm25", "pm10", "pm1"]:
    df = clean_and_cast_column(df, col)

# Remove duplicate rows and sort
df = df.unique().sort('Fecha_Hora')

# Function to get statistics and value occurrences
def get_column_stats(df, col):
    stats = df.select([
        pl.col(col).mean().alias("mean"),
        pl.col(col).median().alias("median"),
        pl.col(col).std().alias("std_dev"),
        pl.col(col).min().alias("min"),
        pl.col(col).max().alias("max"),
    ]).to_dict(as_series=False)
    
    return {
        "statistics": stats,
    }

# Get statistics for each numeric column
stats = {}
for col in ["pm25", "pm10", "pm1"]:
    stats[col] = get_column_stats(df, col)

# Print statistics
for col, col_stats in stats.items():
    print(f"\nStatistics for {col}:")
    print(f"Mean: {col_stats['statistics']['mean'][0]:.2f}")
    print(f"Median: {col_stats['statistics']['median'][0]:.2f}")
    print(f"Standard Deviation: {col_stats['statistics']['std_dev'][0]:.2f}")
    print(f"Min: {col_stats['statistics']['min'][0]:.2f}")
    print(f"Max: {col_stats['statistics']['max'][0]:.2f}")
    print("\nTop 10 most common values and their occurrences:")

# Write the cleaned DataFrame to a new CSV file
df.write_csv("cleaned_data.csv")
