"""
Data loading and preprocessing using MiniCamels
"""
"""
Data loading and preprocessing using MiniCamels
"""
from minicamels import MiniCamels

def summarize_dataset():
    ds = MiniCamels()

    # Basins
    basins_df = ds.basins()
    n_basins = len(basins_df)

    # Static attributes
    attrs_df = ds.attributes()
    n_static = attrs_df.shape[1]

    # Debug: inspect actual column names
    print("Basins columns:", basins_df.columns.tolist())
    print("Attributes columns:", attrs_df.columns.tolist())

    # Use the first column as basin ID
    basin_id_col = basins_df.columns[0]
    example_basin_id = str(basins_df.iloc[0][basin_id_col])

    # Load one basin to inspect variables and time
    ts = ds.load_basin(example_basin_id)

    # Convert to dataframe
    df = ts.to_dataframe().reset_index()
    print("Time series columns:", df.columns.tolist())

    # Find time column safely
    if "time" in df.columns:
        time_col = "time"
    elif "date" in df.columns:
        time_col = "date"
    else:
        time_col = df.columns[0]

    # Time span
    start_date = df[time_col].min()
    end_date = df[time_col].max()

    # Variables
    dynamic_vars = ["prcp", "tmax", "tmin", "srad", "vp"]
    target_var = "qobs"

    print("\n===== MiniCAMELS Dataset Summary =====")
    print(f"Number of basins: {n_basins}")
    print(f"Time span: {start_date} to {end_date}")
    print(f"Dynamic variables: {', '.join(dynamic_vars)}")
    print(f"Target variable: {target_var}")
    print(f"Number of static attributes: {n_static}")
    print(f"Example basin ID column: {basin_id_col}")
    print(f"Example basin ID: {example_basin_id}")
    print("======================================\n")
