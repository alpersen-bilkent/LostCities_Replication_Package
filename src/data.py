# -*- coding: utf-8 -*-
"""
Single source of truth for the raw data used by every experiment:
the ancient 25-city trade matrix and coordinates, the Barjamovic
comparison estimates, the modern 15-city coordinates, and the shared
graph-construction helper.

Previously this block (matrix_data, cities_data, bajramovic_*,
coords_dict, create_graph_data) was copy-pasted, with small silent
drifts, across all five original GCN scripts (see ../legacy/). That
drift is what produced Bug 2 (see src/evaluate.py) and part of the
divergence behind Bug 1 (see src/weights.py).
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from . import config

# ---------------------------------------------------------------------------
# Ancient network: 25 Middle Bronze Age cities (15 known + 10 lost)
# ---------------------------------------------------------------------------

ANCIENT_CITY_NAMES = [
    'Durhumit', 'Hahhum', 'Hanaknak', 'Hattus', 'Hurama', 'Kanes', 'Karahna',
    'Kuburnat', 'Malitta', 'Mamma', 'Ninassa', 'Purushaddum', 'Salatuwar',
    'Samuha', 'Sinahuttum', 'Suppiluliya', 'Tapaggas', 'Timelkiya', 'Tuhpiya',
    'Ulama', 'Unipsum', 'Wahsusana', 'Washaniya', 'Zalpa', 'Zimishuna'
]

# Directed itinerary counts, X_ij = trips FROM row city TO column city.
# Row/column order matches ANCIENT_CITY_NAMES exactly.
ANCIENT_MATRIX_DATA = np.array([
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 13, 5, 0, 1, 0, 0, 1, 2, 0, 0, 10, 0, 0, 1],
    [1, 0, 0, 0, 0, 8, 18, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 19, 0, 0, 1, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 2, 0, 0, 0, 15, 1, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0],
    [0, 7, 0, 0, 0, 0, 2, 1, 2, 0, 3, 3, 2, 0, 0, 0, 0, 4, 2, 2, 0, 12, 5, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0],
    [6, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 3, 0, 0, 0, 7, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 1, 0, 21, 19, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 1, 0, 3, 0, 0, 0, 0, 1, 22, 19, 0, 0, 0, 1, 1, 7, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 1, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
])

# name -> (lat, long); lost cities carry NaN coordinates.
ANCIENT_CITIES = {
    'name': ['Durhumit', 'Hahhum', 'Kuburnat', 'Ninassa', 'Purushaddum', 'Sinahuttum',
             'Suppiluliya', 'Tuhpiya', 'Washaniya', 'Zalpa', 'Hattus', 'Kanes', 'Karahna',
             'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta', 'Salatuwar', 'Samuha', 'Timelkiya',
             'Ulama', 'Unipsum', 'Wahsusana', 'Zimishuna', 'Mamma'],
    'lat_y': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
              40.021, 38.85, 40.0, 40.148, 40.0, 38.261, 39.363, 39.655, 39.619, 38.027,
              38.411, 38.021, 39.584, 40.461, 37.583],
    'long_x': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
               34.61, 35.633, 36.1, 35.762, 35.817, 37.114, 33.787, 31.994, 36.528, 38.234,
               33.834, 36.503, 33.418, 35.65, 36.933],
}

# Barjamovic et al.'s own point estimates for the 15 "known" cities, used only
# as a comparison overlay when those cities are fictitiously held out (LOO).
# Verified against the QJE Supplemental Online Appendix's Table 2 ("Proof of
# Concept (robustness), Recovering One Fictitiously Lost City and Ten Lost
# Cities Jointly") -- matches to within rounding (e.g. Hattus 133.38 vs.
# their 133, Zimishuna 185.64 vs. their 186; mean 116.03 vs. their 116).
# NOT their Appendix Table 1 ("Proof of Concept, Recovering Fictitiously
# Lost Cities", mean 76 km) -- that table fixes the ten genuinely lost
# cities at their own point estimates during each known-city trial, an
# easier setup than Table 2's, where those ten are also re-estimated
# jointly, exactly matching this codebase's own LOO procedure (which never
# fixes the ten lost cities' coordinates either -- see run_loo_experiment()
# in src/evaluate.py). Table 2, not Table 1, is the fair comparison.
BARJAMOVIC_KNOWN_ESTIMATES = {
    'name': ['Hattus', 'Kanes', 'Karahna', 'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta',
             'Mamma', 'Salatuwar', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana',
             'Zimishuna'],
    'est_lat': [39.997, 39.313, 40.046, 40.0, 40.15, 39.139, 38.888, 38.02, 39.561, 38.3,
                38.261, 39.835, 37.583, 39.003, 39.234],
    'est_long': [36.131, 33.918, 34.547, 35.817, 35.761, 38.226, 35.282, 36.503, 33.356,
                 37.118, 37.114, 33.234, 36.933, 31.926, 34.213],
}

# Barjamovic et al.'s STRUCTURAL GRAVITY MODEL point estimate for the 10
# actually-lost cities (their QJE paper's own NLLS output), used as the
# comparison target in the final prediction (paper Table 11/12).
#
# CORRECTED (this session) to TABLE II ("Lost Cities' Geocoordinates", main
# text, p.1475) of the PUBLISHED paper -- Barjamovic, Chaney, Cosar &
# Hortacsu (2019), "Trade, Merchants, and the Lost Cities of the Bronze
# Age," QJE 134(3), doi:10.1093/qje/qjz009 (the exact paper \cite{
# barjamovic2019} refers to). Verified two ways: (1) read directly off
# Table II itself; (2) independently cross-checked against the PUBLISHED
# Online/Supplemental Appendix's own Table 3 ("Assigning Lost Cities to
# Archaeological Sites"), whose per-city "gravity estimate" header repeats
# these same coordinates (e.g. Durhumit 40.47, 35.65 there vs. 40.470,
# 35.650 in Table II) -- two independently-checkable places in the
# published paper agree with each other and with the values below.
#
# History of this constant, for the record: an earlier pass through this
# codebase had it matching Table II (correct), then "corrected" it to
# Appendix Table F.1 of the NBER working paper draft (BarjamovicHarika.pdf,
# WP 23992, Nov 2017) on the theory that Table 3 in the published appendix
# was untrustworthy for this purpose -- that reasoning was wrong (Table 3's
# per-city header IS the model's coordinate estimate; only the *candidate
# site list underneath* is the separate, unrelated thing). Appendix Table
# F.1 is a real table and genuinely is the main/preferred directional-data
# specification (confirmed against its own sibling Table F.2, explicitly
# captioned as the noisier non-directional robustness check) -- it is
# simply an EARLIER, pre-publication draft of the exact same calculation,
# superseded by Table II before the paper was accepted. Comparing against
# the actually-published number is what a citation to the published paper
# should mean, so Table II wins. Confirmed with the corresponding author
# (this session, 2026-08-22) before making this change, since it moves the
# reported "distance to NLLS" figures throughout the paper.
#
# NOT the same thing as BARJAMOVIC_2011_ESTIMATES below, despite the
# shared author name -- that's Barjamovic's own separate, earlier,
# philological/historical-geography conjecture (his 2011 monograph), a
# completely different kind of estimate (expert judgment, not a fitted
# quantitative model) from the same person.
BARJAMOVIC_LOST_ESTIMATES = {
    'name': ['Durhumit', 'Hahhum', 'Kuburnat', 'Ninassa', 'Purushaddum', 'Sinahuttum',
             'Suppiluliya', 'Tuhpiya', 'Washaniya', 'Zalpa'],
    'est_lat': [40.470, 38.429, 40.712, 38.977, 39.710, 39.956, 40.021, 39.611, 39.157, 38.805],
    'est_long': [35.650, 38.040, 36.520, 34.614, 32.872, 34.866, 34.618, 35.199, 34.311, 37.862],
}

# Barjamovic's own 2011 historical-geography monograph proposal for the 10
# lost cities -- philological/archaeological expert judgment, NOT the QJE
# paper's structural gravity model (BARJAMOVIC_LOST_ESTIMATES above).
# Source: user-provided HistoriansAndKnownCities.py, cross-checked against
# raw/ancient/coordinates_forlanini_vs_barjamovic.csv in the replication
# package -- matches to 2 decimal places.
BARJAMOVIC_2011_ESTIMATES = {
    'name': ['Durhumit', 'Hahhum', 'Kuburnat', 'Ninassa', 'Purushaddum', 'Sinahuttum',
             'Suppiluliya', 'Tuhpiya', 'Washaniya', 'Zalpa'],
    'est_lat': [40.88, 37.58, 40.08, 38.95, 38.35, 40.17, 39.68, 39.65, 39.15, 37.79],
    'est_long': [35.46, 38.48, 36.51, 33.55, 31.52, 34.84, 35.48, 33.92, 34.16, 38.62],
}

# Forlanini's 2008 historical-geography proposal for the 10 lost cities --
# same source/verification as BARJAMOVIC_2011_ESTIMATES above.
FORLANINI_2008_ESTIMATES = {
    'name': ['Durhumit', 'Hahhum', 'Kuburnat', 'Ninassa', 'Purushaddum', 'Sinahuttum',
             'Suppiluliya', 'Tuhpiya', 'Washaniya', 'Zalpa'],
    'est_lat': [39.57, 37.58, 40.30, 39.02, 38.40, 40.17, 40.38, 39.33, 38.70, 38.05],
    'est_long': [33.42, 38.48, 35.88, 33.80, 33.83, 34.84, 35.52, 33.78, 34.83, 38.55],
}


def ancient_symmetric_matrix():
    """Undirected trade volume: X_ij + X_ji.

    No longer used by weights.dyadic_weights() (which now takes the RAW
    directed ANCIENT_MATRIX_DATA and self-symmetrizes via its own Eq. 2-3
    cross-term formula -- see src/weights.py) or by
    weights.symmetric_tii_from_directed() (always took the raw matrix).
    Kept for any caller that specifically wants the presymmetrized volume
    itself (e.g. ad-hoc analysis); not used by any src/evaluate.py driver
    as of this revision."""
    return ANCIENT_MATRIX_DATA + ANCIENT_MATRIX_DATA.T


def _ancient_coords_lookup():
    """name -> (long, lat) for the 15 known ancient cities. A lookup by name
    rather than by position, deliberately, so it can never silently drift
    out of sync with ANCIENT_CITY_NAMES order (see note below)."""
    cities_df = pd.DataFrame(ANCIENT_CITIES).dropna()
    return {row['name']: (row['long_x'], row['lat_y']) for _, row in cities_df.iterrows()}


def ancient_known_lost_split():
    """Returns (known_indices, lost_indices, known_names, lost_names), all
    aligned to ANCIENT_CITY_NAMES / ANCIENT_MATRIX_DATA order -- i.e. graph
    node order.

    Undocumented bug found during the refactor (not in HANDOFF.md's list of
    6): the original legacy/GCNDyadicBootstrapResults.py computed these
    indices from `cities_data`'s own listing order (Durhumit, Hahhum,
    Kuburnat, ... , Mamma) via `np.where(cities_df['lat_y'].notna())`, then
    used those same integer positions to index `out`, the model's node
    output -- but `out` is ordered by `all_city_names`/`matrix_data`
    (Durhumit, Hahhum, Hanaknak, Hattus, ...), a *different* permutation.
    The two orderings only agree on the first two names, so the training
    loss silently paired 8 known cities' predictions (plus 7 lost cities'
    predictions) against the wrong target coordinates for the entire
    200-run bootstrap that produced paper Table 11. This is likely a
    larger source of error in the current lost-city predictions than
    Bug 1. known_indices/known_coords/known_names are now derived here
    from a single name-keyed lookup, so they cannot drift apart."""
    coords_lookup = _ancient_coords_lookup()
    known_indices = np.array([i for i, name in enumerate(ANCIENT_CITY_NAMES) if name in coords_lookup])
    lost_indices = np.array([i for i, name in enumerate(ANCIENT_CITY_NAMES) if name not in coords_lookup])
    known_names = [ANCIENT_CITY_NAMES[i] for i in known_indices]
    lost_names = [ANCIENT_CITY_NAMES[i] for i in lost_indices]
    return known_indices, lost_indices, known_names, lost_names


def ancient_known_coords():
    """(N_known, 2) array of [long, lat], row-aligned with the known_names
    returned by ancient_known_lost_split() (graph/matrix node order)."""
    coords_lookup = _ancient_coords_lookup()
    _, _, known_names, _ = ancient_known_lost_split()
    coords = np.array([coords_lookup[name] for name in known_names])
    return coords, known_names


def barjamovic_known_lookup():
    """name -> (long, lat) for Barjamovic et al.'s point estimates of the 15
    known cities, used as a comparison overlay in fictitious-LOO plots."""
    df = pd.DataFrame(BARJAMOVIC_KNOWN_ESTIMATES).set_index('name')
    return {name: (df.loc[name, 'est_long'], df.loc[name, 'est_lat']) for name in df.index}


def barjamovic_2011_lookup():
    """name -> (long, lat) for Barjamovic's own 2011 historical-geography
    monograph proposal for the 10 lost cities (philological judgment, NOT
    the QJE paper's structural gravity model -- see BARJAMOVIC_2011_ESTIMATES'
    docstring)."""
    df = pd.DataFrame(BARJAMOVIC_2011_ESTIMATES).set_index('name')
    return {name: (df.loc[name, 'est_long'], df.loc[name, 'est_lat']) for name in df.index}


def forlanini_2008_lookup():
    """name -> (long, lat) for Forlanini's 2008 historical-geography
    proposal for the 10 lost cities."""
    df = pd.DataFrame(FORLANINI_2008_ESTIMATES).set_index('name')
    return {name: (df.loc[name, 'est_long'], df.loc[name, 'est_lat']) for name in df.index}


def barjamovic_lost_lookup():
    """name -> (long, lat) for Barjamovic et al.'s point estimates of the 10
    actually-lost cities, used as the comparison target in the final
    prediction (paper Table 11/12)."""
    df = pd.DataFrame(BARJAMOVIC_LOST_ESTIMATES).set_index('name')
    return {name: (df.loc[name, 'est_long'], df.loc[name, 'est_lat']) for name in df.index}


# Named archaeological sites mentioned in Barjamovic et al.'s Supplemental
# Appendix Table 3 ("Assigning Lost Cities to Archaeological Sites"). Real,
# sourced, named sites with real coordinates -- the closest thing to a
# registered-site inventory available for this project.
#
# Deliberately flattened and city-agnostic: their table pairs each site
# with a specific lost city (by proximity to THEIR structural-gravity
# estimate for that city), but reusing that pairing to check our own
# model's ellipse would be circular -- it would just be testing whether
# we agree with a conclusion we imported, not doing independent
# proximity screening. src/terrain.py's hoyuk_proximity_score() looks up
# the nearest site in this whole list, regardless of which lost city
# Barjamovic originally paired it with; a site can legitimately be a
# good candidate for whichever lost city it ends up geographically near,
# not only the one they assigned it to. Deduplicated (several sites were
# a top-5 candidate for more than one lost city in their table).
KNOWN_ARCHAEOLOGICAL_SITES = [
    # (name, lat, lon)
    ('Ayvalıpınar', 40.46, 35.65),
    ('Oluz Höyük', 40.55, 35.63),
    ('Doğantepe', 40.6, 35.6),
    ('Ferzant', 40.6, 35.38),
    ('Boyalı', 40.31, 34.26),
    ('Imikuşağı', 38.52, 38.46),
    ('Değirmentepe', 38.48, 38.45),
    ('Imamoğlu', 38.48, 38.48),
    ('Arslantepe', 38.38, 38.36),
    ('Yassıhöyük (Tanır)', 38.39, 36.91),
    ('Tekkeköy (Samsun)', 41.2, 36.45),
    ('Dündartepe', 41.25, 36.35),
    ('Kaledoruğu (Kavak)', 41.08, 36.04),
    ('Kayapınar Höyüğü', 40.16, 36.25),
    ('Bolus (Aktepe)', 40.07, 36.5),
    ('Suluca Karahöyük (Hacıbektaş)', 38.93, 34.55),
    ('Topakhöyük', 38.61, 34.29),
    ('Zank', 38.95, 34.79),
    ('Topaklı', 39.01, 34.83),
    ('Uşaklı/Kuşaklı Höyük', 39.8, 35.1),
    ('Karaoğlan', 39.73, 32.83),
    ('Külhöyük (Haymana)', 39.48, 32.67),
    ('Ballıkuyumcu', 39.77, 32.52),
    ('Çomaklı/İlmez', 37.72, 32.5),
    ('Ortakaraviran II', 37.38, 32.09),
    ('Yassıhöyük (Yozgat)', 39.99, 34.88),
    ('Çengeltepe', 39.84, 34.87),
    ('Eskiyapar', 40.16, 34.77),
    ('Mercimektepe', 40.88, 35.34),
    ('Alaca Höyük', 40.23, 34.68),
    ('Büyüknefes (Bronze Age Site)', 39.85, 34.5),
    ('Çadır (Sorgun)', 39.68, 35.14),
    ('Boğazlıyan/Yoğunhisar', 39.17, 35.23),
    ('Üyük', 40.15, 35.85),
    ('Yassıhöyük (Çoğun/Kırşehir)', 39.32, 34.08),
    ('Harmandalı', 38.95, 33.95),
    ('Yalak (Boz Höyük)', 38.3, 36.44),
    ('Sarız', 38.47, 36.5),
    # Not from Barjamovic et al.'s Table 3 -- added separately. A major,
    # continuously excavated site (Turkish Ministry-recognized excavation
    # 1962-present: Nimet Özgüç 1962-88, Aliye Öztan since 1989).
    # Coordinates: Wikipedia (en.wikipedia.org/wiki/Acemhöyük), cross-
    # checked against the Megalithic Portal. Proposed (NOT confirmed) as
    # the site of Old Assyrian Purušḫattum/Purushanda -- i.e. this
    # project's lost city Purushaddum -- by Joost Blasweiler (2019); this
    # is a live scholarly debate, not consensus. Note the coincidence:
    # Forlanini's own 2008 coordinate estimate for Purushaddum
    # (FORLANINI_2008_ESTIMATES below, 38.40 / 33.83) lands almost
    # exactly on this site independently of Blasweiler's later argument.
    # Kept city-agnostic in this list like every other entry above --
    # screening code never sees or uses the Purushaddum identification,
    # only the coordinates.
    ('Acemhöyük', 38.4116, 33.8355),
]


# ---------------------------------------------------------------------------
# Modern network: 15 Central/Eastern Anatolian province centers
# ---------------------------------------------------------------------------

MODERN_CITY_COORDS = {
    'ADIYAMAN': (37.7648, 38.2786), 'AKSARAY': (38.3687, 34.0370),
    'AMASYA': (40.6500, 35.8300), 'ÇANKIRI': (40.6013, 33.6134),
    'ÇORUM': (40.5506, 34.9556), 'KAHRAMANMARAŞ': (37.5710, 36.9371),
    'KAYSERİ': (38.7312, 35.4787), 'KIRIKKALE': (39.8468, 33.5153),
    'KIRŞEHİR': (39.1425, 34.1709), 'MALATYA': (38.3552, 38.3095),
    'NEVŞEHİR': (38.6244, 34.7144), 'NİĞDE': (37.9667, 34.6833),
    'SİVAS': (39.7477, 37.0179), 'TOKAT': (40.3167, 36.5500),
    'YOZGAT': (39.8181, 34.8147),
}


def modern_cities_and_coords():
    """Returns (sorted_city_names, (N,2) array of [long, lat])."""
    cities_df = pd.DataFrame.from_dict(MODERN_CITY_COORDS, orient='index',
                                        columns=['lat_y', 'long_x'])
    valid_cities = sorted(cities_df.index.tolist())
    known_coords = cities_df.loc[valid_cities, ['long_x', 'lat_y']].values
    return valid_cities, known_coords


def _load_modern_trade_dataframe(file_path=None, valid_cities=None):
    file_path = file_path or config.MODERN_TRADE_FILE
    if valid_cities is None:
        valid_cities, _ = modern_cities_and_coords()

    df_raw = pd.read_excel(file_path, header=0)
    df_raw.rename(columns={df_raw.columns[0]: 'City'}, inplace=True)
    df_raw.set_index('City', inplace=True)
    df_raw = df_raw.replace({'*': 0, '-': 0}).infer_objects(copy=False)
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_raw.loc[valid_cities, valid_cities]


def load_modern_matrix(file_path=None, valid_cities=None):
    """Loads İller Arası Ticaret.xlsx and returns the RAW (directed,
    generally asymmetric -- exports_ij need not equal exports_ji) trade
    matrix restricted to valid_cities, in that order. Used by
    weights.dyadic_weights() (paper Eq. 2-3, which symmetrizes via its own
    cross-term formula and needs the directional X_ij/X_ji values to do
    so) and weights.symmetric_tii_from_directed()."""
    df_subset = _load_modern_trade_dataframe(file_path, valid_cities)
    return df_subset.values.astype(np.float32)


def load_modern_symmetric_matrix(file_path=None, valid_cities=None):
    """Loads İller Arası Ticaret.xlsx and returns the PRESYMMETRIZED
    (X_ij + X_ji) trade matrix restricted to valid_cities, in that order.
    Used by weights.symmetric_tii_from_symmetric() / tii_weights_from_
    symmetric(), which require an already-undirected input (they do not
    self-symmetrize the way the directed variants do -- see weights.py)."""
    df_subset = _load_modern_trade_dataframe(file_path, valid_cities)
    return (df_subset + df_subset.T).values.astype(np.float32)


# ---------------------------------------------------------------------------
# Shared graph construction
# ---------------------------------------------------------------------------

def create_graph_data(weight_matrix, num_nodes):
    """Builds a torch_geometric Data object from a weighted adjacency matrix.
    Previously duplicated (identically) across all five original scripts."""
    rows, cols = np.where(weight_matrix > 0)
    edge_indices = list(zip(rows.tolist(), cols.tolist()))
    edge_weights = weight_matrix[rows, cols]
    return Data(
        x=torch.arange(num_nodes),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_weights, dtype=torch.float),
    )
