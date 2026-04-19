"""Tests for io.py helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from minicamels.io import (
    _open_remote_dataset,
    _remote_csv_url,
    _remote_timeseries_url,
    open_basin_dataset,
)
from tests.conftest import BASIN_IDS


def test_remote_timeseries_url():
    url = _remote_timeseries_url("01013500")
    assert url.endswith("/timeseries/01013500.nc")
    assert url.startswith("https://")


def test_remote_csv_url():
    url = _remote_csv_url("basins.csv")
    assert url.endswith("/basins.csv")


def test_open_basin_dataset_local(data_dir):
    ds = open_basin_dataset(BASIN_IDS[0], local_data_dir=data_dir)
    assert "prcp" in ds
    assert "time" in ds.dims


def test_open_basin_dataset_missing_local(data_dir):
    with pytest.raises(FileNotFoundError):
        open_basin_dataset("00000000", local_data_dir=data_dir)


def test_open_remote_dataset_fallback(tmp_path):
    """NetCDF4-only files should load even when scipy backend fails."""
    ds = xr.Dataset(
        data_vars={"foo": ("time", np.arange(3, dtype="float32"))},
        coords={"time": np.arange(3)},
    )
    nc_path = tmp_path / "foo.nc"
    ds.to_netcdf(nc_path, engine="netcdf4")

    opened = _open_remote_dataset(nc_path.as_uri())
    xr.testing.assert_equal(opened, ds)
    opened.close()
