"""Runtime settings, all overridable through environment variables."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))          # the VM's ~/dblp, mounted read-only
PARQUET = Path(os.environ.get("PARQUET", DATA_DIR / "parquet" / "dblp.parquet"))
EDA_OUT = Path(os.environ.get("EDA_OUT", DATA_DIR / "eda_out"))  # text output of the analysis jobs
CHARTS_DIR = Path(os.environ.get("CHARTS_DIR", DATA_DIR / "charts"))  # CSVs behind the report's charts
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache"))        # writable: serving database + spill files

DUCKDB_MEMORY = os.environ.get("DUCKDB_MEMORY", "12GB")
DUCKDB_THREADS = int(os.environ.get("DUCKDB_THREADS", "6"))
WATCH_SECONDS = int(os.environ.get("WATCH_SECONDS", "600"))    # how often to look for a new parquet
HEAVY_QUERY_SLOTS = int(os.environ.get("HEAVY_QUERY_SLOTS", "3"))  # concurrent full-table scans allowed

# Bump when the serving-table SQL changes: forces a rebuild even if the parquet did not change.
BUILD_VERSION = 1
