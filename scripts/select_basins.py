"""
Select 50 CAMELS-US basins that maximize climate diversity.

Usage
-----
    python scripts/select_basins.py --camels-dir /path/to/camels_us/ \
        [--n-basins 50] [--seed 42] [--output scripts/selected_basins.txt]

The script writes a plain text file with one 8-digit USGS gauge ID per line.
Commit both this script and selected_basins.txt so the selection is
fully reproducible without requiring the raw CAMELS files.

Algorithm
---------
1. Load CAMELS attributes (topology, climate, hydrology).
2. Pre-filter for data quality (area bounds, missing-data threshold).
3. Assign Köppen-Geiger macro bins (A, B, C, D_cont, D_snow, E).
4. Allocate 50 slots proportionally to US land area, floor of 3 per bin.
5. Within each bin, pick basins via maximin-distance sampling on
   (lat, lon, aridity) to spread selection geographically and climatically.
6. Write selected_basins.txt.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# KG macro-bin mapping
# ---------------------------------------------------------------------------

# Maps individual KG codes (as they appear in CAMELS) to 6 macro bins.
# CAMELS uses a simplified set of codes; unmapped codes fall back to the
# first character of the code.
KG_BIN_MAP: dict[str, str] = {
    # Tropical
    "Af": "A", "Am": "A", "Aw": "A", "As": "A",
    # Arid / Semi-arid
    "BWh": "B", "BWk": "B", "BSh": "B", "BSk": "B",
    # Temperate (Mediterranean + oceanic + humid subtropical)
    "Cfa": "C", "Cfb": "C", "Cfc": "C",
    "Csa": "C", "Csb": "C", "Csc": "C",
    "Cwa": "C", "Cwb": "C",
    # Continental humid
    "Dfa": "D_cont", "Dfb": "D_cont",
    "Dwa": "D_cont", "Dwb": "D_cont",
    # Continental subarctic / snow
    "Dfc": "D_snow", "Dfd": "D_snow",
    "Dwc": "D_snow", "Dwd": "D_snow",
    "Dsc": "D_snow", "Dsb": "D_snow",
    # Polar / Alpine
    "ET": "E", "EF": "E",
}

# Target number of basins per macro bin (must sum to n_basins).
# Proportions reflect approximate US land-area coverage by each KG class.
BIN_QUOTAS: dict[str, int] = {
    "B":      12,
    "C":      14,
    "D_cont": 10,
    "D_snow":  8,
    "A":       3,
    "E":       3,
}
assert sum(BIN_QUOTAS.values()) == 50, "BIN_QUOTAS must sum to 50"


# ---------------------------------------------------------------------------
# Attribute loading
# ---------------------------------------------------------------------------

def _load_attributes(camels_dir: Path) -> pd.DataFrame:
    """Load and merge CAMELS attribute tables into a single DataFrame."""
    attr_dir = camels_dir / "camels_attributes_v2.0"
    if not attr_dir.exists():
        # Try alternate layout
        attr_dir = camels_dir / "attributes"

    tables = {}
    for fname in attr_dir.glob("camels_*.txt"):
        df = pd.read_csv(fname, sep=";", dtype={"gauge_id": str})
        df = df.rename(columns={"gauge_id": "basin_id"})
        df = df.set_index("basin_id")
        tables[fname.stem] = df

    if not tables:
        raise FileNotFoundError(
            f"No camels_*.txt attribute files found in {attr_dir}"
        )

    merged = pd.concat(tables.values(), axis=1)
    # Drop duplicate columns that appear in multiple files
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


# ---------------------------------------------------------------------------
# Pre-filtering
# ---------------------------------------------------------------------------

def _prefilter(attrs: pd.DataFrame) -> pd.DataFrame:
    """Remove basins with data quality issues or extreme sizes."""
    n_start = len(attrs)
    mask = pd.Series(True, index=attrs.index)

    if "area_gages2" in attrs.columns:
        mask &= attrs["area_gages2"].between(50, 10_000)
    if "q_mean" in attrs.columns:
        mask &= attrs["q_mean"].notna() & (attrs["q_mean"] > 0)

    attrs = attrs[mask]
    print(f"Pre-filter: {n_start} → {len(attrs)} basins")
    return attrs


# ---------------------------------------------------------------------------
# KG bin assignment
# ---------------------------------------------------------------------------

def _assign_kg_bin(attrs: pd.DataFrame) -> pd.DataFrame:
    """Add a 'kg_bin' column derived from CAMELS climate attributes.

    CAMELS-US does not ship KG codes directly. Bins are derived from
    aridity, frac_snow, and latitude using Budyko-style thresholds.
    """
    attrs = attrs.copy()

    def _proxy_bin(row):
        a = row.get("aridity", float("nan"))
        s = row.get("frac_snow", float("nan"))
        lat = row.get("gauge_lat", float("nan"))
        elev = row.get("elev_mean", 0.0)

        # Use NaN-safe comparisons (NaN comparisons return False)
        if not (a == a):  # a is NaN
            return "C"

        # Arid / semi-arid: PET > P
        if a > 1.0:
            return "B"

        # Humid basins (aridity <= 1.0) — classify by snow fraction
        if not (s == s):  # s is NaN
            s = 0.0

        # Alpine / polar proxy: very high snow + high elevation or far north
        if s > 0.6 and (elev > 2000 or (lat == lat and lat > 46)):
            return "E"
        # Continental subarctic / heavy-snow
        if s > 0.4:
            return "D_snow"
        # Continental humid
        if s > 0.1:
            return "D_cont"
        # Subtropical / tropical margins (warm, low latitude)
        if (lat == lat) and lat < 33:
            return "A"
        # Temperate (Mediterranean, oceanic, humid subtropical)
        return "C"

    attrs["kg_bin"] = attrs.apply(_proxy_bin, axis=1)
    print("KG bin distribution (proxy from aridity + frac_snow + lat):")
    print(attrs["kg_bin"].value_counts().to_string())
    return attrs


# ---------------------------------------------------------------------------
# Maximin distance sampling
# ---------------------------------------------------------------------------

def _maximin_sample(
    candidates: pd.DataFrame,
    n: int,
    features: list[str],
    rng: np.random.Generator,
) -> list[str]:
    """
    Greedy farthest-point (maximin) sampling in a standardized feature space.

    Returns a list of basin_ids of length ``min(n, len(candidates))``.
    """
    if len(candidates) == 0:
        return []
    n = min(n, len(candidates))

    feat = candidates[features].copy()
    # Drop rows with any NaN in the feature columns
    feat = feat.dropna()
    if len(feat) == 0:
        return candidates.index[:n].tolist()

    # Standardize
    feat_std = (feat - feat.mean()) / (feat.std() + 1e-9)
    X = feat_std.values  # (m, d)
    ids = feat.index.tolist()

    seed_idx = int(rng.integers(len(ids)))
    selected_indices = [seed_idx]

    while len(selected_indices) < n:
        selected_X = X[selected_indices]  # (k, d)
        # Distance from each candidate to its nearest selected point
        diffs = X[:, None, :] - selected_X[None, :, :]  # (m, k, d)
        dists = np.sqrt((diffs ** 2).sum(axis=-1))        # (m, k)
        min_dists = dists.min(axis=1)                     # (m,)
        # Pick the candidate with the largest min distance
        next_idx = int(np.argmax(min_dists))
        if next_idx in selected_indices:
            # Fallback: pick a random unchosen candidate
            remaining = list(set(range(len(ids))) - set(selected_indices))
            if not remaining:
                break
            next_idx = int(rng.choice(remaining))
        selected_indices.append(next_idx)

    return [ids[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Main selection
# ---------------------------------------------------------------------------

def select_basins(
    attrs: pd.DataFrame,
    n_basins: int = 50,
    seed: int = 42,
) -> list[str]:
    """
    Select ``n_basins`` basins maximizing climate diversity.

    Returns a sorted list of basin IDs.
    """
    rng = np.random.default_rng(seed)
    attrs = _prefilter(attrs)
    attrs = _assign_kg_bin(attrs)

    # Adjust quotas if n_basins != 50
    if n_basins != 50:
        scale = n_basins / 50
        quotas = {k: max(1, round(v * scale)) for k, v in BIN_QUOTAS.items()}
        # Reconcile rounding
        diff = n_basins - sum(quotas.values())
        for k in sorted(quotas, key=lambda x: -BIN_QUOTAS[x]):
            if diff == 0:
                break
            quotas[k] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
    else:
        quotas = BIN_QUOTAS.copy()

    feature_cols = [c for c in ("gauge_lat", "gauge_lon", "aridity") if c in attrs.columns]
    if not feature_cols:
        feature_cols = [attrs.columns[0]]

    selected: list[str] = []
    for bin_name, quota in quotas.items():
        group = attrs[attrs["kg_bin"] == bin_name]
        if len(group) == 0:
            print(f"  Warning: no basins found for bin '{bin_name}' — skipping.")
            continue
        chosen = _maximin_sample(group, quota, feature_cols, rng)
        print(f"  {bin_name:8s}: {len(chosen):2d} / {quota} selected "
              f"(from {len(group)} candidates)")
        selected.extend(chosen)

    print(f"\nTotal selected: {len(selected)} basins")
    return sorted(selected)


# ---------------------------------------------------------------------------
# CLI
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
    parser.add_argument("--n-basins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "selected_basins.txt",
        help="Output text file (one basin_id per line).",
    )
    args = parser.parse_args()

    attrs = _load_attributes(args.camels_dir)
    print(f"Loaded attributes for {len(attrs)} CAMELS basins.")

    basin_ids = select_basins(attrs, n_basins=args.n_basins, seed=args.seed)

    args.output.write_text("\n".join(basin_ids) + "\n")
    print(f"Wrote {len(basin_ids)} basin IDs to {args.output}")


if __name__ == "__main__":
    main()
