"""
Build the minicamels data directory from raw CAMELS-US files.

Usage
-----
    python scripts/prepare_data.py \
        --camels-dir /path/to/camels_us/ \
        --basin-list scripts/selected_basins.txt \
        [--output-dir data/] \
        [--overwrite]

Expected raw CAMELS layout
--------------------------
camels_us/
├── basin_timeseries_v1p2_metForcing_obsFlow/
│   └── basin_dataset_public_v1p2/
│       └── basin_mean_forcing/
│           └── daymet/
│               └── {huc02}/
│                   └── {basin_id}_lump_cida_forcing_leap.txt
├── usgs_streamflow/
│   └── {huc02}/
│       └── {basin_id}_streamflow_qc.txt
└── camels_attributes_v2.0/
    ├── camels_clim.txt
    ├── camels_geol.txt
    ├── camels_hydro.txt
    ├── camels_name.txt
    ├── camels_soil.txt
    ├── camels_topo.txt
    └── camels_vege.txt

All forcings are Daymet-derived. The script does not process Maurer or
NLDAS forcing variants.
"""

from __future__ import annotations

import argparse
import textwrap
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATE_START = "1980-10-01"
DATE_END = "2010-09-30"
TARGET_INDEX = pd.date_range(DATE_START, DATE_END, freq="D")

# NetCDF variable metadata: name -> (long_name, units)
VAR_ATTRS = {
    "prcp": ("Daymet precipitation",             "mm/day"),
    "tmax": ("Daymet daily max air temperature", "degC"),
    "tmin": ("Daymet daily min air temperature", "degC"),
    "srad": ("Daymet shortwave radiation",        "W/m2"),
    "vp":   ("Daymet vapor pressure",             "Pa"),
    "qobs": ("Observed streamflow (USGS)",        "mm/day"),
}

NC_ENCODING = {
    "time": {
        "dtype": "int32",
        "units": f"days since {DATE_START}",
        "calendar": "standard",
    },
    **{
        v: {"dtype": "float32", "zlib": True, "complevel": 4, "_FillValue": 9.96921e36}
        for v in VAR_ATTRS
    },
}

# Attributes columns to retain in attributes.csv
ATTR_COLUMNS = [
    "basin_id",
    "gauge_lat", "gauge_lon", "elev_mean", "slope_mean", "area_gages2",
    "p_mean", "pet_mean", "aridity", "frac_snow",
    "high_prcp_freq", "low_prcp_freq",
    "q_mean", "runoff_ratio", "hfd_mean", "baseflow_index",
    "soil_depth_pelletier", "frac_forest", "lai_max",
]

# Rename from CAMELS column names to friendlier names for students
ATTR_RENAME = {
    "gauge_lat": "lat",
    "gauge_lon": "lon",
    "area_gages2": "area_km2",
    "p_mean": "mean_prcp",
    "pet_mean": "mean_pet",
}


# ---------------------------------------------------------------------------
# Attribute loading
# ---------------------------------------------------------------------------

def _load_all_attributes(camels_dir: Path) -> pd.DataFrame:
    """Merge all camels_*.txt attribute files into a single DataFrame."""
    attr_dir = _find_attr_dir(camels_dir)
    dfs = []
    for fpath in sorted(attr_dir.glob("camels_*.txt")):
        df = pd.read_csv(fpath, sep=";", dtype={"gauge_id": str})
        df = df.rename(columns={"gauge_id": "basin_id"})
        dfs.append(df.set_index("basin_id"))

    merged = pd.concat(dfs, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged.index.name = "basin_id"
    return merged


def _find_attr_dir(camels_dir: Path) -> Path:
    for candidate in [
        camels_dir / "camels_attributes_v2.0",
        camels_dir / "attributes",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find attribute directory under {camels_dir}. "
        "Expected 'camels_attributes_v2.0/' or 'attributes/'."
    )


# ---------------------------------------------------------------------------
# Forcing loading
# ---------------------------------------------------------------------------

def _find_forcing_file(camels_dir: Path, basin_id: str) -> Path:
    """Locate the Daymet forcing file for one basin.

    The CAMELS forcing files are organized by USGS HUC-2 watershed region,
    which does not correspond to the first two digits of the gauge ID.
    Search all HUC subdirectories.
    """
    for root in [
        camels_dir / "basin_mean_forcing" / "daymet",
        camels_dir / "basin_timeseries_v1p2_metForcing_obsFlow"
        / "basin_dataset_public_v1p2" / "basin_mean_forcing" / "daymet",
        camels_dir / "forcing" / "daymet",
        camels_dir / "daymet",
    ]:
        if not root.exists():
            continue
        for match in root.rglob(f"{basin_id}_*.txt"):
            return match
    raise FileNotFoundError(
        f"Daymet forcing file not found for basin {basin_id}. "
        f"Searched under {camels_dir}."
    )


def _read_forcing(path: Path) -> pd.DataFrame:
    """
    Read a CAMELS Daymet forcing file.

    The file has 3 metadata lines (lat, area, id), then a column header line,
    then tab/space-delimited daily data:
    Year  Mnth  Day  Hr  dayl(s)  prcp(mm/day)  srad(W/m2)  swe(mm)  tmax(C)  tmin(C)  vp(Pa)
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=4,  # 3 metadata lines + 1 column header line
        header=None,
        names=["year", "month", "day", "hour", "dayl", "prcp", "srad", "swe",
               "tmax", "tmin", "vp"],
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]].rename(
        columns={"year": "year", "month": "month", "day": "day"}
    ))
    df = df.set_index("date")[["prcp", "srad", "tmax", "tmin", "vp"]]
    return df


# ---------------------------------------------------------------------------
# Streamflow loading
# ---------------------------------------------------------------------------

def _find_streamflow_file(camels_dir: Path, basin_id: str) -> Path:
    for root in [
        camels_dir / "usgs_streamflow",
        camels_dir / "streamflow",
    ]:
        if not root.exists():
            continue
        for match in root.rglob(f"{basin_id}_*.txt"):
            return match
    raise FileNotFoundError(
        f"Streamflow file not found for basin {basin_id}. "
        f"Searched under {camels_dir}."
    )


def _read_streamflow(path: Path, area_km2: float) -> pd.Series:
    """
    Read CAMELS streamflow file and convert to mm/day.

    Raw columns: basin_id  Year  Mnth  Day  qobs_cfs  qc_flag
    Conversion: cfs -> mm/day using catchment area.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["basin_id", "year", "month", "day", "qobs_cfs", "qc_flag"],
        dtype={"basin_id": str, "qc_flag": str},
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]].rename(
        columns={"year": "year", "month": "month", "day": "day"}
    ))
    df = df.set_index("date")

    # Convert ft3/s -> mm/day
    # q [mm/day] = q [ft3/s] * 0.0283168 [m3/ft3] * 86400 [s/day]
    #              / (area_km2 * 1e6 [m2/km2]) * 1000 [mm/m]
    q = df["qobs_cfs"].astype(float)
    q_mm = q * 0.0283168 * 86400 / (area_km2 * 1e6) * 1000

    # Mask missing / negative values
    q_mm = q_mm.where(df["qc_flag"].str.strip() != "M")
    q_mm = q_mm.where(q >= 0)

    return q_mm.rename("qobs")


# ---------------------------------------------------------------------------
# Per-basin NetCDF writer
# ---------------------------------------------------------------------------

def _build_dataset(
    forcing: pd.DataFrame,
    streamflow: pd.Series,
    basin_id: str,
    attrs_row: pd.Series,
    basin_name: str,
) -> xr.Dataset:
    """Align forcing and streamflow, build and return an xr.Dataset."""
    # Reindex both to the canonical daily index; forward-fill at most 1 day
    # to handle leap-day artefacts in Daymet files.
    forcing = forcing.reindex(TARGET_INDEX)
    n_missing_forcing = forcing.isna().any(axis=1).sum()
    if n_missing_forcing > 0:
        warnings.warn(
            f"{basin_id}: {n_missing_forcing} missing forcing days after reindex — "
            "forward-filling up to 1 day."
        )
        forcing = forcing.ffill(limit=1)

    streamflow = streamflow.reindex(TARGET_INDEX)

    # Build Dataset
    coords = {"time": TARGET_INDEX}
    data_vars = {}
    for var in ["prcp", "tmax", "tmin", "srad", "vp"]:
        long_name, units = VAR_ATTRS[var]
        data_vars[var] = xr.DataArray(
            forcing[var].values.astype("float32"),
            dims=["time"],
            attrs={"long_name": long_name, "units": units},
        )

    long_name, units = VAR_ATTRS["qobs"]
    data_vars["qobs"] = xr.DataArray(
        streamflow.values.astype("float32"),
        dims=["time"],
        attrs={
            "long_name": long_name,
            "units": units,
            "source": "USGS",
        },
    )

    ds = xr.Dataset(data_vars, coords=coords)
    ds["time"].attrs["long_name"] = "time"

    # Global attributes
    def _safe(val):
        """Convert numpy scalars to native Python types for NetCDF."""
        if hasattr(val, "item"):
            return val.item()
        return val

    ds.attrs = {
        "Conventions": "CF-1.8",
        "basin_id": basin_id,
        "basin_name": basin_name,
        "area_km2": _safe(attrs_row.get("area_gages2", float("nan"))),
        "lat": _safe(attrs_row.get("gauge_lat", float("nan"))),
        "lon": _safe(attrs_row.get("gauge_lon", float("nan"))),
        "huc02": basin_id[:2],
        "forcing": "Daymet",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": "CAMELS-US (Newman et al. 2015; Addor et al. 2017)",
    }

    return ds


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_basins_csv(
    basin_ids: list[str],
    attrs: pd.DataFrame,
    output_dir: Path,
):
    names = []
    for bid in basin_ids:
        if bid in attrs.index:
            name = attrs.loc[bid].get("gauge_name", "")
        else:
            name = ""
        names.append(str(name))

    df = pd.DataFrame({"basin_id": basin_ids, "basin_name": names})
    df.to_csv(output_dir / "basins.csv", index=False)
    print(f"  Wrote {output_dir / 'basins.csv'}")


def _write_attributes_csv(
    basin_ids: list[str],
    attrs: pd.DataFrame,
    output_dir: Path,
):
    available = [c for c in ATTR_COLUMNS if c in ["basin_id"] or c in attrs.columns]
    subset = attrs.loc[attrs.index.isin(basin_ids), [c for c in available if c != "basin_id"]]
    subset.index.name = "basin_id"
    subset = subset.rename(columns=ATTR_RENAME)
    subset.to_csv(output_dir / "attributes.csv")
    print(f"  Wrote {output_dir / 'attributes.csv'}  ({len(subset)} basins × {len(subset.columns)} attrs)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(output_dir: Path, basin_ids: list[str]) -> bool:
    from io import StringIO

    print("\nValidation pass...")
    issues = []
    for bid in basin_ids:
        path = output_dir / "timeseries" / f"{bid}.nc"
        try:
            ds = xr.open_dataset(path, engine="netcdf4")
        except Exception as e:
            issues.append(f"  {bid}: cannot open — {e}")
            continue

        if len(ds.time) != len(TARGET_INDEX):
            issues.append(f"  {bid}: time length {len(ds.time)}, expected {len(TARGET_INDEX)}")

        for var in ["prcp", "tmax", "tmin", "srad", "vp", "qobs"]:
            if var not in ds:
                issues.append(f"  {bid}: missing variable '{var}'")

        ds.close()

    if issues:
        print("Issues found:")
        for msg in issues:
            print(msg)
        return False

    # Missing-data summary table
    rows = []
    for bid in basin_ids:
        ds = xr.open_dataset(output_dir / "timeseries" / f"{bid}.nc", engine="netcdf4")
        row = {"basin_id": bid}
        for var in ["prcp", "tmax", "tmin", "srad", "vp", "qobs"]:
            pct = float(np.isnan(ds[var].values).mean() * 100)
            row[var] = f"{pct:.1f}%"
        rows.append(row)
        ds.close()

    summary = pd.DataFrame(rows).set_index("basin_id")
    buf = StringIO()
    summary.to_csv(buf)
    print("\nMissing data summary (% NaN per variable):")
    print(summary.to_string())
    print("\nAll files passed validation.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--camels-dir",
        required=True,
        type=Path,
        help="Root directory of the CAMELS-US download.",
    )
    parser.add_argument(
        "--basin-list",
        required=True,
        type=Path,
        help="Text file with one basin_id per line (output of select_basins.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Output data/ directory (default: repo data/).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .nc files.",
    )
    args = parser.parse_args()

    basin_ids = [
        line.strip()
        for line in args.basin_list.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Processing {len(basin_ids)} basins → {args.output_dir}")

    ts_dir = args.output_dir / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    # Load attributes once
    print("Loading CAMELS attributes...")
    attrs = _load_all_attributes(args.camels_dir)

    # Write CSV files first
    _write_basins_csv(basin_ids, attrs, args.output_dir)
    _write_attributes_csv(basin_ids, attrs, args.output_dir)

    # Process each basin
    errors = []
    for i, basin_id in enumerate(basin_ids, 1):
        out_path = ts_dir / f"{basin_id}.nc"
        if out_path.exists() and not args.overwrite:
            print(f"  [{i:2d}/{len(basin_ids)}] {basin_id}  skipped (exists)")
            continue

        try:
            attrs_row = attrs.loc[basin_id] if basin_id in attrs.index else pd.Series()
            area_km2 = float(attrs_row.get("area_gages2", 1000.0))
            basin_name = str(attrs_row.get("gauge_name", ""))

            forcing = _read_forcing(_find_forcing_file(args.camels_dir, basin_id))
            streamflow = _read_streamflow(
                _find_streamflow_file(args.camels_dir, basin_id), area_km2
            )

            ds = _build_dataset(forcing, streamflow, basin_id, attrs_row, basin_name)
            ds.to_netcdf(out_path, encoding=NC_ENCODING, engine="netcdf4")

            size_kb = out_path.stat().st_size / 1024
            print(f"  [{i:2d}/{len(basin_ids)}] {basin_id}  {size_kb:.0f} KB")

        except Exception as exc:
            errors.append((basin_id, exc))
            print(f"  [{i:2d}/{len(basin_ids)}] {basin_id}  ERROR: {exc}")

    print(f"\nDone. {len(basin_ids) - len(errors)} succeeded, {len(errors)} failed.")
    if errors:
        for bid, exc in errors:
            print(f"  {bid}: {exc}")

    _validate(args.output_dir, [b for b, _ in errors if False] or basin_ids)


if __name__ == "__main__":
    main()
