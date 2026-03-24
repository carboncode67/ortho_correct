"""
Orthomosaic Spatial Offset Correction
--------------------------------------
Walks TIFF_DIR for ortho TIFs, GCP_DIR for uncorrected GCP CSVs,
reads CORRECTED_CSV for corrected positions, then writes offset-corrected
TIFs to OUTPUT_DIR.

Edit the five paths and four column names below before running.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.spatial import cKDTree
from pyproj import Transformer

# ---------------------------------------------------------------------------
# CONFIGURE THESE
# ---------------------------------------------------------------------------

TIFF_DIR        = Path(r"C:\path\to\ortho_tifs")          # walks subdirs
GCP_DIR         = Path(r"C:\path\to\gcp_csvs")            # walks subdirs
CORRECTED_CSV   = Path(r"C:\path\to\corrected_points.csv")
OUTPUT_DIR      = Path(r"C:\path\to\output")

# Column names in CORRECTED_CSV
CORRECTED_X_COL = "X"
CORRECTED_Y_COL = "Y"

# Column names in each GCP CSV (uncorrected position, read from first data row)
GCP_LAT_COL = "Base latitude"
GCP_LON_COL = "Base longitude"

# Max distance (metres) for point-to-point matching
POINT_MATCH_THRESHOLD = 5.0
# Max distance (metres) for ortho centroid-to-corrected-point matching
ORTHO_MATCH_THRESHOLD = 100.0

# ---------------------------------------------------------------------------


def find_files(root: Path, extensions: tuple) -> list:
    """Walk root recursively and return all files matching extensions."""
    return [p for p in root.rglob("*") if p.suffix.lower() in extensions]


def load_corrected_points(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in (CORRECTED_X_COL, CORRECTED_Y_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"Corrected CSV missing columns: {missing}  (found: {list(df.columns)})")
    df = df[[CORRECTED_X_COL, CORRECTED_Y_COL]].dropna().copy()
    df.columns = ["x", "y"]
    df["point_id"] = range(len(df))
    print(f"  Loaded {len(df)} corrected points from {csv_path.name}")
    return df


def get_tif_crs(tiff_dir: Path) -> str:
    """Return the CRS string of the first TIF found under tiff_dir."""
    for p in tiff_dir.rglob("*"):
        if p.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(p) as src:
                if src.crs is None:
                    raise ValueError(f"TIF has no CRS: {p}")
                print(f"  Detected TIF CRS: {src.crs} (from {p.name})")
                return str(src.crs)
    raise FileNotFoundError(f"No TIF files found under {tiff_dir} to detect CRS")


def load_gcp_points(gcp_dir: Path, tif_crs: str) -> pd.DataFrame:
    """Load GCP CSVs and reproject lat/lon to the TIF's projected CRS."""
    transformer = Transformer.from_crs("EPSG:4326", tif_crs, always_xy=True)

    csv_files = find_files(gcp_dir, (".csv",))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {gcp_dir}")
    rows = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=1)
            lat = df[GCP_LAT_COL].iloc[0]
            lon = df[GCP_LON_COL].iloc[0]
            if pd.isna(lat) or pd.isna(lon):
                print(f"  SKIP {f.name}: NaN coordinates")
                continue
            x, y = transformer.transform(float(lon), float(lat))
            rows.append({"x": x, "y": y, "source": f.name})
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
    if not rows:
        raise ValueError("No valid GCP points could be read")
    result = pd.DataFrame(rows)
    result["point_id"] = range(len(result))
    print(f"  Loaded {len(result)} GCP points from {len(csv_files)} CSV files")
    return result


def match_points(uncorrected: pd.DataFrame, corrected: pd.DataFrame) -> dict:
    """
    Match each uncorrected GCP to its nearest corrected point.
    Returns dict mapping corrected point_id -> (x_offset, y_offset).
    """
    tree = cKDTree(corrected[["x", "y"]].values)
    distances, indices = tree.query(uncorrected[["x", "y"]].values)

    offsets = {}
    matched = 0
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist > POINT_MATCH_THRESHOLD:
            print(f"  UNMATCHED GCP {uncorrected.iloc[i]['source']}: "
                  f"nearest corrected point is {dist:.2f}m away (threshold {POINT_MATCH_THRESHOLD}m)")
            continue
        corr = corrected.iloc[idx]
        uncorr = uncorrected.iloc[i]
        x_off = corr["x"] - uncorr["x"]
        y_off = corr["y"] - uncorr["y"]
        offsets[int(corr["point_id"])] = (x_off, y_off)
        matched += 1
        print(f"  Matched {uncorr['source']} -> corrected pt {int(corr['point_id'])} "
              f"(dist {dist:.2f}m, offset {x_off:+.3f}, {y_off:+.3f})")

    print(f"  Point matching: {matched}/{len(uncorrected)} matched")
    return offsets


def get_tiff_centroid(tiff_path: Path):
    with rasterio.open(tiff_path) as src:
        b = src.bounds
        return (b.left + b.right) / 2, (b.bottom + b.top) / 2


def apply_offset(input_path: Path, output_path: Path, x_off: float, y_off: float):
    with rasterio.open(input_path) as src:
        t = src.transform
        new_transform = Affine(t.a, t.b, t.c + x_off,
                               t.d, t.e, t.f + y_off)
        meta = src.meta.copy()
        meta["transform"] = new_transform
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(src.read())


def main():
    print("\n=== Orthomosaic Offset Correction ===\n")

    # 1. Load points
    print("Loading corrected points...")
    corrected = load_corrected_points(CORRECTED_CSV)

    print("Detecting TIF CRS...")
    tif_crs = get_tif_crs(TIFF_DIR)

    print("Loading uncorrected GCP points...")
    uncorrected = load_gcp_points(GCP_DIR, tif_crs)

    # 2. Match points and compute offsets
    print("Matching GCP points to corrected positions...")
    offsets = match_points(uncorrected, corrected)
    if not offsets:
        raise RuntimeError("No GCP points matched. Check paths, column names, and threshold.")

    # 3. Find TIFs
    print(f"\nDiscovering TIFs under {TIFF_DIR} ...")
    tiff_files = find_files(TIFF_DIR, (".tif", ".tiff"))
    print(f"  Found {len(tiff_files)} TIF files")
    if not tiff_files:
        raise FileNotFoundError(f"No TIF files found under {TIFF_DIR}")

    # 4. Match each TIF centroid to nearest corrected point, apply offset
    print(f"\nProcessing TIFs -> {OUTPUT_DIR}\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tree = cKDTree(corrected[["x", "y"]].values)
    processed = skipped = 0

    for tif in tiff_files:
        try:
            cx, cy = get_tiff_centroid(tif)
            dist, idx = tree.query([cx, cy])

            if dist > ORTHO_MATCH_THRESHOLD:
                print(f"  SKIP {tif.name}: centroid {dist:.1f}m from nearest corrected point "
                      f"(threshold {ORTHO_MATCH_THRESHOLD}m)")
                skipped += 1
                continue

            corr_id = int(corrected.iloc[idx]["point_id"])
            if corr_id not in offsets:
                print(f"  SKIP {tif.name}: matched corrected pt {corr_id} has no GCP offset")
                skipped += 1
                continue

            x_off, y_off = offsets[corr_id]
            out_path = OUTPUT_DIR / (tif.stem + "_corrected" + tif.suffix)
            apply_offset(tif, out_path, x_off, y_off)
            print(f"  OK  {tif.name}  offset ({x_off:+.3f}, {y_off:+.3f})  -> {out_path.name}")
            processed += 1

        except Exception as e:
            print(f"  ERROR {tif.name}: {e}")
            skipped += 1

    print(f"\nDone. {processed} corrected, {skipped} skipped/failed.")


if __name__ == "__main__":
    main()
