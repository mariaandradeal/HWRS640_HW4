"""
Shared fixtures for minicamels tests.

Tests use a tiny synthetic dataset (2 basins, 365 days) written to a
temporary directory so no real CAMELS data is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from minicamels._constants import NC_ENCODING, VAR_ATTRS


BASIN_IDS = ["01013500", "02084500"]
DATE_INDEX = pd.date_range("1980-10-01", periods=365, freq="D")


def _make_synthetic_dataset(basin_id: str) -> xr.Dataset:
    rng = np.random.default_rng(int(basin_id))
    n = len(DATE_INDEX)
    data_vars = {}
    for var, (long_name, units) in VAR_ATTRS.items():
        values = rng.random(n).astype("float32")
        data_vars[var] = xr.DataArray(
            values,
            dims=["time"],
            attrs={"long_name": long_name, "units": units},
        )
    ds = xr.Dataset(data_vars, coords={"time": DATE_INDEX})
    ds.attrs = {
        "Conventions": "CF-1.8",
        "basin_id": basin_id,
        "basin_name": f"Test basin {basin_id}",
        "area_km2": 1000.0,
        "lat": 45.0,
        "lon": -90.0,
        "huc02": basin_id[:2],
        "forcing": "Daymet",
    }
    return ds


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """Temporary data directory with synthetic NetCDF and CSV files."""
    root = tmp_path_factory.mktemp("data")
    ts_dir = root / "timeseries"
    ts_dir.mkdir()

    # Encoding subset — only time + vars present in the short dataset
    short_encoding = {
        "time": {"dtype": "int32", "units": "days since 1980-10-01", "calendar": "standard"},
        **{
            v: {"dtype": "float32", "zlib": True, "complevel": 1, "_FillValue": 9.96921e36}
            for v in VAR_ATTRS
        },
    }

    for bid in BASIN_IDS:
        ds = _make_synthetic_dataset(bid)
        ds.to_netcdf(ts_dir / f"{bid}.nc", encoding=short_encoding, engine="netcdf4")

    # basins.csv
    pd.DataFrame(
        {"basin_id": BASIN_IDS, "basin_name": [f"Test {b}" for b in BASIN_IDS]}
    ).to_csv(root / "basins.csv", index=False)

    # attributes.csv
    pd.DataFrame(
        {
            "basin_id": BASIN_IDS,
            "lat": [45.0, 35.0],
            "lon": [-90.0, -80.0],
            "area_km2": [1000.0, 500.0],
            "aridity": [0.8, 1.5],
            "q_mean": [1.2, 0.6],
        }
    ).to_csv(root / "attributes.csv", index=False)

    return root
