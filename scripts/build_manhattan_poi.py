"""
Build data/geo/replica-manhattan-poi.csv (POI_FILE) from the 4 weekly Advan parquet files.

Required output schema: exactly what POISearchTool reads (mobility_agent/tools.py:44-157) -
LOCATION_NAME, TOP_CATEGORY, SUB_CATEGORY, NAICS_CODE, LATITUDE, LONGITUDE, RAW_VISIT_COUNTS.
No NAICS-to-trip-purpose crosswalk needed here (that's a different exercise) - the LLM/agent
decides which NAICS code(s) to search for, this file is just the raw POI catalog with
popularity, like safegraph-cambridge-poi2.csv was for Cambridge.

Key decisions (verified against the actual parquet schema, not assumed):
- PERSISTENT_ID looked like a per-POI id but turned out to be a brand/category-level
  fallback (only 1406 unique values for 151k Manhattan rows, e.g. "A1_NAICS-722513").
  ID_STORE (== PERSISTENT_ID_STORE) is the real per-location id: 38,203 unique Manhattan
  POIs, 0 nulls.
- Filtered to Manhattan via POI_CBG starting with "36061" (New York County FIPS), since the
  raw Advan pull covers all of New York state (REGION=NY was chosen deliberately over a
  City="New York" text filter, which was unreliable - see project notes).
- VISIT_COUNTS averaged across however many of the 4 weekly files each POI appears in (not
  all POIs appear in all 4 weeks). CAVEAT: the 2025-11-24 week includes Thanksgiving
  (Nov 27), an atypical week for many categories (restaurants/travel up, offices/schools
  down) - averaging it in with the other 3 weeks is a known, documented limitation, not
  something this script tries to correct.
- Renamed VISIT_COUNTS -> RAW_VISIT_COUNTS to match the column name tools.py expects.
"""
import glob

import pandas as pd

PARQUET_FILES = sorted(glob.glob("data/geo/advan_manhattan_raw/*.parquet"))
OUT_PATH = "data/geo/replica-manhattan-poi.csv"

COLUMNS = [
    "ID_STORE", "LOCATION_NAME", "TOP_CATEGORY", "SUB_CATEGORY",
    "NAICS_CODE", "LATITUDE", "LONGITUDE", "POI_CBG", "VISIT_COUNTS",
]

frames = [pd.read_parquet(f, columns=COLUMNS) for f in PARQUET_FILES]
all_df = pd.concat(frames, ignore_index=True)
print(f"loaded {len(all_df)} rows across {len(PARQUET_FILES)} weekly files")

manhattan = all_df[all_df["POI_CBG"].astype(str).str.startswith("36061")].copy()
print(f"filtered to {len(manhattan)} rows in Manhattan (POI_CBG startswith 36061)")

agg = manhattan.groupby("ID_STORE").agg({
    "LOCATION_NAME": "first",
    "TOP_CATEGORY": "first",
    "SUB_CATEGORY": "first",
    "NAICS_CODE": "first",
    "LATITUDE": "first",
    "LONGITUDE": "first",
    "VISIT_COUNTS": "mean",
}).reset_index(drop=True)

agg = agg.rename(columns={"VISIT_COUNTS": "RAW_VISIT_COUNTS"})
agg["RAW_VISIT_COUNTS"] = agg["RAW_VISIT_COUNTS"].round(1)

# tools.py reads this via geopandas.read_file() -> GDAL's CSV driver, which on Windows
# defaults to the system codepage (cp1252) unless it sees a UTF-8 BOM. Some LOCATION_NAME
# values have mis-encoded characters (from the source Advan data, not introduced here) that
# aren't valid cp1252 at all and crash the reader - utf-8-sig adds the BOM GDAL auto-detects.
agg.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"wrote {len(agg)} unique Manhattan POIs to {OUT_PATH}")
print(agg["TOP_CATEGORY"].value_counts().head(10))
