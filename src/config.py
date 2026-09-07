# -*- coding: utf-8 -*-
"""
Paths and global constants shared across every experiment.

Paths are read from environment variables with the current machine's
paths as defaults, so the repo still runs out of the box here but can
be pointed elsewhere (e.g. a reviewer's machine) via env vars instead
of editing source.
"""

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
THIRD_PARTY_DIR = os.path.join(PROJECT_ROOT, "third_party_data")

RESULTS_DIR = os.environ.get("GCN_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
FIGURES_DIR = os.environ.get("GCN_FIGURES_DIR", os.path.join(RESULTS_DIR, "figures"))

# All three of the paths below default to this package's own bundled copy
# (third_party_data/, checked in alongside the code precisely so this repo
# runs out of the box on any machine, not just the one it was built on) --
# env vars remain available to point at a different/updated copy instead.
SHAPEFILE_PATH = os.environ.get(
    "GCN_SHAPEFILE_PATH",
    os.path.join(THIRD_PARTY_DIR, "turkey_shapefile", "gadm41_TUR_1.shp"),
)

MODERN_TRADE_FILE = os.environ.get(
    "GCN_MODERN_TRADE_FILE",
    os.path.join(THIRD_PARTY_DIR, "modern_trade_data", "İller Arası Ticaret.xlsx"),
)

# Barjamovic et al.'s own replication package -- source of the §7.3 terrain
# inputs (elevation, rivers, precomputed road-knot scores). See src/terrain.py.
# Only the specific files this codebase actually reads are bundled (a few
# hundred KB) -- not their full ~2GB package, most of which (other papers'
# figures, unused rasters, scanned map sheets) this pipeline never touches.
REPLICATION_PACKAGE_DIR = os.environ.get(
    "GCN_REPLICATION_PACKAGE_DIR",
    os.path.join(THIRD_PARTY_DIR, "barjamovic_replication_package"),
)
ELEVATION_TIF = os.path.join(
    REPLICATION_PACKAGE_DIR, "figures_tables", "GEOdata", "fao_gaez_elevation", "faodata-clipped.tif",
)
RIVERS_SHP = os.path.join(
    REPLICATION_PACKAGE_DIR, "roadknots", "Input", "rivers", "ne_10m_rivers_lake_centerlines.shp",
)
RIVERS_EUROPE_SHP = os.path.join(
    REPLICATION_PACKAGE_DIR, "roadknots", "Input", "rivers", "ne_10m_rivers_europe.shp",
)
LAKES_SHP = os.path.join(REPLICATION_PACKAGE_DIR, "roadknots", "Input", "rivers", "lakes_turkey.shp")
CROSSING_COORD_XLSX = os.path.join(REPLICATION_PACKAGE_DIR, "roadknots", "Input", "crossing_coord.xlsx")

# Known Early-Bronze-Age-era mineral deposits (copper/silver/gold/tin),
# compiled by Barjamovic et al. from De Jesus (1980) and Massa (2016) --
# same replication package, informational overlay only (see
# src/terrain.py:load_mineral_deposits()). Not used for scoring or
# exclusion, since Barjamovic et al.'s own regression found distance to
# copper deposits not significant/robust (their Table 4 notes).
MINERAL_DEPOSITS_DTA = os.path.join(
    REPLICATION_PACKAGE_DIR, "figures_tables", "GEOdata", "ancient_mineral_deposits", "ancientminedata.dta",
)

# Default screening bounding box: the same Central/Eastern Anatolia range
# already used throughout plots.py (xlim=(31, 39.3), ylim=(35.8, 42.2)).
SCREENING_LON_RANGE = (31.0, 39.3)
SCREENING_LAT_RANGE = (35.8, 42.2)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MASTER_SEED = 42
LATITUDE_PARAM = 37.9
KM_PER_DEGREE = 10000.0 / 90.0

# Barjamovic, Chaney, Cosar & Hortacsu (2019)'s own headline out-of-sample
# error for their structural gravity model -- the number this paper's
# LOO results (ancient_loo_dyadic.py / ancient_loo_tii.py) are compared
# against. Published as a single aggregate figure, not broken down city
# by city, which is why the significance test in src/evaluate.py is a
# one-sample bootstrap test against this fixed value rather than a
# paired test.
BARJAMOVIC_LOO_HEADLINE_KM = 116.03
