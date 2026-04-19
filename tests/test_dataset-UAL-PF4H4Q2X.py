"""Tests for MiniCamels dataset class (local data only)."""

from __future__ import annotations

import pytest
import xarray as xr
import pandas as pd

from minicamels import MiniCamels
from tests.conftest import BASIN_IDS


@pytest.fixture
def ds(data_dir):
    return MiniCamels(local_data_dir=data_dir)


def test_repr(ds):
    r = repr(ds)
    assert "MiniCamels" in r
    assert "Daymet" in r


def test_basins_returns_dataframe(ds):
    df = ds.basins()
    assert isinstance(df, pd.DataFrame)
    assert "basin_id" in df.columns
    assert len(df) == len(BASIN_IDS)


def test_attributes_returns_dataframe(ds):
    df = ds.attributes()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "basin_id"
    assert "lat" in df.columns


def test_open_basin_returns_dataset(ds):
    result = ds.open_basin(BASIN_IDS[0])
    assert isinstance(result, xr.Dataset)
    for var in ["prcp", "tmax", "tmin", "srad", "vp", "qobs"]:
        assert var in result


def test_load_basin_is_loaded(ds):
    result = ds.load_basin(BASIN_IDS[0])
    # After load(), all variables should be numpy-backed (no lazy chunks)
    assert result["prcp"].values is not None


def test_open_basins_concat(ds):
    result = ds.open_basins(BASIN_IDS)
    assert "basin" in result.dims
    assert result.sizes["basin"] == len(BASIN_IDS)


def test_get_forcings_excludes_qobs(ds):
    result = ds.get_forcings(BASIN_IDS[0])
    assert "qobs" not in result
    assert "prcp" in result


def test_get_streamflow_returns_dataarray(ds):
    result = ds.get_streamflow(BASIN_IDS[0])
    assert isinstance(result, xr.DataArray)
    assert result.name == "qobs"


def test_get_water_year_length(ds):
    result = ds.get_water_year(BASIN_IDS[0], water_year=1981)
    # WY1981 = 1980-10-01 to 1981-09-30 = 365 days
    assert len(result.time) == 365


def test_get_water_year_out_of_range(ds):
    with pytest.raises(ValueError, match="water_year"):
        ds.get_water_year(BASIN_IDS[0], water_year=1970)


def test_get_forcings_time_slice(ds):
    result = ds.get_forcings(BASIN_IDS[0], start="1980-10-01", end="1980-10-31")
    assert len(result.time) == 31


def test_missing_basin_raises(ds):
    with pytest.raises(FileNotFoundError):
        ds.open_basin("99999999")
