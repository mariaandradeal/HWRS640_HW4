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

    # Load one basin to inspect variables and time
    example_basin_id = basins_df.iloc[0]["gauge_id"]
    ts = ds.load_basin(example_basin_id)

    # Convert to dataframe
    df = ts.to_dataframe().reset_index()

    # Time span
    start_date = df["time"].min()
    end_date = df["time"].max()

    # Variables
    dynamic_vars = ["prcp", "tmax", "tmin", "srad", "vp"]
    target_var = "qobs"

    print("\n===== MiniCAMELS Dataset Summary =====")
    print(f"Number of basins: {n_basins}")
    print(f"Time span: {start_date} to {end_date}")
    print(f"Dynamic variables: {', '.join(dynamic_vars)}")
    print(f"Target variable: {target_var}")
    print(f"Number of static attributes: {n_static}")
    print("======================================\n")
