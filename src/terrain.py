# -*- coding: utf-8 -*-
"""
§7.3 post-hoc screening: rank candidate cells inside a lost city's
confidence ellipse by terrain and water. Every input here comes
directly from Barjamovic et al.'s own replication package
(config.REPLICATION_PACKAGE_DIR) rather than a re-download or a
reimplementation:

  - Elevation/slope: figures_tables/GEOdata/fao_gaez_elevation/faodata-clipped.tif
                      -- their own FAO-GAEZ 5 arc-min grid, already clipped
                      to Turkey, EPSG:4326.
  - Rivers/lakes:     roadknots/Input/rivers/*.shp (their own copy of
                      Natural Earth + a Turkey lakes layer).
  - River crossings:  roadknots/Input/crossing_coord.xlsx (loaded for
                      completeness; not used by screen_ellipse() below --
                      see load_river_crossings()).
  - Known sites:      data.KNOWN_ARCHAEOLOGICAL_SITES -- 38 named sites
                      transcribed from Barjamovic et al.'s Supplemental
                      Appendix Table 3 "Assigning Lost Cities to
                      Archaeological Sites" (qjz009_supplemental_appendix.pdf).
                      Real named sites, real coordinates -- not a full
                      registry (still no TAY Project / Ministry of Culture
                      data), but real and sourced. IMPORTANT: kept
                      deliberately city-agnostic, NOT paired with the lost
                      city Barjamovic et al. originally assigned each site
                      to -- see data.KNOWN_ARCHAEOLOGICAL_SITES' docstring
                      for why reusing their pairing would be circular.
  - Mineral deposits: figures_tables/GEOdata/ancient_mineral_deposits/
                      ancientminedata.dta -- their own compilation of 66
                      Early-Bronze-Age copper/silver/gold/tin deposits,
                      sourced from Massa (2016), "Networks before Empires:
                      cultural transfers in west and central Anatolia
                      during the Early Bronze Age" (unpublished PhD
                      dissertation, UCL) -- confirmed directly against
                      Barjamovic et al.'s own paper text (footnote 25 and
                      the Conclusion both cite Massa 2016 by name for this
                      exact mine list) and its own References list, which
                      does NOT contain a "De Jesus 1980" entry at all. An
                      earlier version of this docstring attributed the data
                      to "De Jesus 1980 and Massa 2016" -- that De Jesus
                      attribution was never verified and is not correct;
                      removed. Informational overlay only (see
                      minerals_within_polygon() below): NOT used to score,
                      rank, or exclude any region, because Barjamovic et
                      al.'s own regression (their Table 4 notes) found
                      distance to copper deposits not significant or
                      robust as a predictor.

Removed: an earlier version of this module also scored candidates by
Barjamovic et al.'s precomputed road-knot (betweenness-centrality) grid
(roadknots/Output/roadknots.csv). It was never used in rank_score (its
own docstring flagged it as unverified -- known cities scored LOWER on
it than random points, the opposite of Barjamovic et al.'s own reported
finding, likely because the single column present in this copy of their
package is one distance-threshold slice rather than the full
multi-threshold aggregate their own pipeline sums, and the intermediate
files needed to reconstruct that aggregate aren't in this copy of the
package). Its raw values were also so small (~1e-7 to 1e-11) that they
displayed as 0.000 at any normal rounding, reading as a data bug even
though the underlying numbers were real. Given it was unverified,
already excluded from every reported number, and effectively
unreadable, it was removed outright rather than kept around unused.

Simplification vs. the original pipeline: roadknots/Code/rivers.py
manually clips five named rivers (Seyhan, Firat, Yesilirmak, Sakarya,
Porsuk) to specific tributary segments before rasterizing them, because
their downstream use is PATH PASSABILITY (can a route cross the river
here). Our use here is a simple screening penalty (distance to the
nearest river/lake, not a passability check), where that segment-level
precision matters far less, so distance_to_water_km() below uses the
raw, unclipped river/lake geometries. If a real cost-surface/Dijkstra
step is ever built on top of this, port rivers.py's clipping logic
exactly rather than reuse this simplification.

All distances use the same simplified km-per-degree conversion
(config.KM_PER_DEGREE, config.LATITUDE_PARAM) as the rest of this
codebase, for consistency with the reported LOO/prediction errors --
not a rigorous geodesic distance.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_origin
from shapely.geometry import Point
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union
from shapely.prepared import prep as prepared_geom

from . import config, data

_WATER_CACHE = None
_WATER_UNION_CACHE = None
_LAKE_CACHE = None
_LAKE_UNION_CACHE = None
_MINERALS_CACHE = None


def sample_elevation(lons, lats):
    """Elevation (m) at each (lon, lat), nearest-cell sample from the
    FAO-GAEZ grid.

    Deliberately does NOT use rasterio's src.sample() generator, which
    calls into rasterio.transform.rowcol() / rasterio.sample.sample_gen()
    -- a per-point Cython/C code path that was crashing with "Fatal
    Python error: Aborted" specifically inside a Spyder/IPython kernel on
    Windows (reproduced reliably there, never in a plain script
    invocation, even after a full kernel restart -- a Spyder-specific
    interaction with rasterio's per-point sampling internals, not a data
    or logic problem). This reads the elevation band once and indexes it
    with the raster's own affine transform directly, in plain NumPy --
    verified to produce bit-identical results to the old method on real
    coordinates before switching over. Also faster: one array read
    instead of one native call per point."""
    lons, lats = np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    with rasterio.open(config.ELEVATION_TIF) as src:
        band = src.read(1)
        inv_transform = ~src.transform
        cols, rows = inv_transform * (lons, lats)
        rows = np.clip(np.floor(rows).astype(int), 0, band.shape[0] - 1)
        cols = np.clip(np.floor(cols).astype(int), 0, band.shape[1] - 1)
        vals = band[rows, cols].astype(float)
        nodata = src.nodata
    if nodata is not None:
        vals[vals == nodata] = np.nan
    return vals


def sample_slope(lons, lats, delta_deg=None):
    """Approximate slope (degrees) via a finite difference on the
    elevation grid, one cell wide by default."""
    lons, lats = np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    with rasterio.open(config.ELEVATION_TIF) as src:
        step = delta_deg or src.res[0]

    cos_ref = np.cos(np.radians(config.LATITUDE_PARAM))
    e_here = sample_elevation(lons, lats)
    e_east = sample_elevation(lons + step, lats)
    e_north = sample_elevation(lons, lats + step)

    dx_m = config.KM_PER_DEGREE * cos_ref * step * 1000.0
    dy_m = config.KM_PER_DEGREE * step * 1000.0
    d_elev_dx = (e_east - e_here) / dx_m
    d_elev_dy = (e_north - e_here) / dy_m
    return np.degrees(np.arctan(np.sqrt(d_elev_dx ** 2 + d_elev_dy ** 2)))


def load_water_geometries(force_reload=False):
    """Rivers + lakes as one GeoSeries (unclipped -- see module docstring)."""
    global _WATER_CACHE
    if _WATER_CACHE is None or force_reload:
        lakes = gpd.read_file(config.LAKES_SHP)
        rivers1 = gpd.read_file(config.RIVERS_SHP)
        rivers2 = gpd.read_file(config.RIVERS_EUROPE_SHP)
        combined = pd.concat([rivers1.geometry, rivers2.geometry, lakes.geometry], ignore_index=True)
        _WATER_CACHE = gpd.GeoSeries(combined, crs=rivers1.crs)
    return _WATER_CACHE


def load_lake_geometries(force_reload=False):
    """Lakes ONLY (config.LAKES_SHP), separate from load_water_geometries()'s
    combined rivers+lakes union -- needed for in_lake_mask() below, which
    treats lakes and rivers differently (a lake's basin, e.g. Tuz Golu, is
    a stable, permanent exclusion; a river's exact channel migrates over
    centuries, so being near or on today's mapped course is weak evidence
    against an ancient settlement having been there -- see
    in_lake_mask()'s docstring)."""
    global _LAKE_CACHE
    if _LAKE_CACHE is None or force_reload:
        lakes = gpd.read_file(config.LAKES_SHP)
        _LAKE_CACHE = gpd.GeoSeries(lakes.geometry, crs=lakes.crs)
    return _LAKE_CACHE


def in_lake_mask(lons, lats, lakes=None, buffer_km=0.2):
    """Boolean mask: which (lon, lat) points fall on/inside a mapped lake
    (buffered by `buffer_km`, matching geo_screened_region()'s own
    river_buffer_km convention) -- e.g. Tuz Golu, in range for several
    lost cities' confidence ellipses.

    Deliberately lake-only, not the combined river+lake water layer
    screen_ellipse()'s dist_water_km ranking already uses: a large salt
    lake like Tuz Golu is barren, seasonally flooded, and has stayed in
    roughly the same basin for millennia, so a candidate cell sitting on
    it is a real settlement-siting exclusion regardless of era. A river's
    mapped course, by contrast, can shift channels over centuries, so a
    candidate being close to or even on today's river line is weak
    evidence against an ancient settlement having been nearby -- rivers
    are left to the ordinary continuous dist_water_km ranking (closer is
    still scored as generally favorable, for irrigation/drinking access),
    not this hard exclusion."""
    global _LAKE_UNION_CACHE
    lakes = lakes if lakes is not None else load_lake_geometries()
    if _LAKE_UNION_CACHE is None:
        _LAKE_UNION_CACHE = lakes.union_all() if hasattr(lakes, 'union_all') else lakes.unary_union
    buffer_deg = buffer_km / config.KM_PER_DEGREE
    exclusion = prepared_geom(_LAKE_UNION_CACHE.buffer(buffer_deg))
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    return np.array([exclusion.contains(Point(lon, lat)) for lon, lat in zip(lons, lats)])


def distance_to_water_km(lons, lats, water=None):
    """Approximate distance (km) from each point to the nearest river/lake
    geometry.

    Uses a single merged shapely geometry (cached) and shapely's own
    .distance() directly, instead of GeoSeries.distance() against every
    individual river/lake geometry per point. Two problems with the old
    approach: it was O(n_points x n_water_geometries), and
    GeoSeries.distance() re-triggers geopandas's geographic-CRS
    UserWarning on every single call via a slow inspect.stack()-based
    message-formatting path -- with 100+ grid points per city, that
    warning machinery (not the actual distance math) was the dominant
    cost, severely so inside a Spyder kernel, which was not deduplicating
    the repeated identical warning the way a plain script's default
    warning filter does, making a real screening run look hung. Distance
    to the union of geometries is mathematically identical to the
    minimum distance to each individual geometry -- verified bit-for-bit
    equal to the old method on real coordinates before switching over --
    and plain shapely objects don't carry GeoSeries's CRS-check wrapper,
    so the warning never fires at all."""
    global _WATER_UNION_CACHE
    water = water if water is not None else load_water_geometries()
    if _WATER_UNION_CACHE is None:
        _WATER_UNION_CACHE = water.union_all() if hasattr(water, 'union_all') else water.unary_union
    cos_ref = np.cos(np.radians(config.LATITUDE_PARAM))
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    dists_deg = np.array([_WATER_UNION_CACHE.distance(Point(lon, lat)) for lon, lat in zip(lons, lats)])
    return dists_deg * config.KM_PER_DEGREE * cos_ref


def load_river_crossings():
    """Fords/crossing points from Barjamovic et al.'s package. Not used by
    screen_ellipse() below (that's point screening, not a passability
    computation) -- kept for a future cost-surface/Dijkstra step."""
    sheet1 = pd.read_excel(config.CROSSING_COORD_XLSX, sheet_name='Sheet1')
    sheet2 = pd.read_excel(config.CROSSING_COORD_XLSX, sheet_name='Sheet2')
    pts1 = sheet1[['_X', '_Y']].dropna().rename(columns={'_X': 'lon', '_Y': 'lat'})
    pts2 = sheet2[['crossing_X', 'crossing_Y']].dropna().rename(columns={'crossing_X': 'lon', 'crossing_Y': 'lat'})
    return pd.concat([pts1, pts2], ignore_index=True).reset_index(drop=True)


def hoyuk_proximity_score(lons, lats, sites=None):
    """Distance (km) from each point to the nearest known archaeological
    site in data.KNOWN_ARCHAEOLOGICAL_SITES (or a caller-supplied `sites`
    list of (name, lat, lon) triples). Lower is better.

    Deliberately city-agnostic -- see data.KNOWN_ARCHAEOLOGICAL_SITES'
    docstring. This never looks up "the sites assigned to city X"; it
    just asks "is there a known site near this candidate point at all,"
    regardless of which lost city (if any) Barjamovic et al. paired that
    site with."""
    sites = sites if sites is not None else data.KNOWN_ARCHAEOLOGICAL_SITES
    site_lats = np.array([s[1] for s in sites])
    site_lons = np.array([s[2] for s in sites])
    cos_ref = np.cos(np.radians(config.LATITUDE_PARAM))

    lons = np.asarray(lons, dtype=float)[:, None]
    lats = np.asarray(lats, dtype=float)[:, None]
    dlon_km = (lons - site_lons[None, :]) * cos_ref * config.KM_PER_DEGREE
    dlat_km = (lats - site_lats[None, :]) * config.KM_PER_DEGREE
    return np.sqrt(dlon_km ** 2 + dlat_km ** 2).min(axis=1)


def _mahalanobis_mask(lons, lats, mean_long, mean_lat, cov, n_std):
    """Boolean mask: which (lon, lat) points fall inside the TRUE,
    possibly-rotated n_std confidence ellipse implied by the full 2x2
    covariance matrix `cov` (as produced by evaluate.summarize_bootstrap()'s
    Var_Long/Var_Lat/Cov_LonLat columns) -- the same Mahalanobis-distance
    test geo_screened_region() already uses, exposed here so the
    axis-aligned callers below (sites_within_ellipse(), sample_ellipse_grid())
    can optionally use the exact same ellipse instead of the
    mean+/-std approximation that ignores any correlation between a city's
    longitude and latitude errors across bootstrap runs."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    cov_inv = np.linalg.inv(np.asarray(cov, dtype=float))
    d = np.column_stack([lons - mean_long, lats - mean_lat])
    mahal2 = np.einsum('ij,jk,ik->i', d, cov_inv, d)
    return mahal2 <= n_std ** 2


def sites_within_ellipse(mean_long, mean_lat, std_long, std_lat, n_std=2.0, sites=None, cov=None):
    """Which known archaeological sites (data.KNOWN_ARCHAEOLOGICAL_SITES,
    or a caller-supplied `sites` list) fall geometrically inside this
    n_std confidence ellipse. City-agnostic: checks the same full pool
    against any ellipse, rather than looking up sites "belonging" to a
    particular city -- see data.KNOWN_ARCHAEOLOGICAL_SITES' docstring for
    why. A site can land inside more than one city's ellipse if those
    ellipses overlap; that's reported as-is, not resolved in favor of
    either city.

    cov: optional 2x2 covariance matrix (e.g. built from a results row's
    Var_Long/Var_Lat/Cov_LonLat). If given, containment uses the TRUE,
    possibly-rotated Mahalanobis ellipse instead of the axis-aligned
    mean+/-std approximation below -- std_long/std_lat are still used for
    dist_to_mean_km and are otherwise ignored for the containment test
    itself in that case.

    Returns a DataFrame (site, lat, lon, dist_to_mean_km), empty if none
    fall inside."""
    sites = sites if sites is not None else data.KNOWN_ARCHAEOLOGICAL_SITES
    cos_ref = np.cos(np.radians(config.LATITUDE_PARAM))
    if cov is not None:
        site_lons = np.array([s[2] for s in sites])
        site_lats = np.array([s[1] for s in sites])
        inside_mask = _mahalanobis_mask(site_lons, site_lats, mean_long, mean_lat, cov, n_std)
    else:
        inside_mask = None
    rows = []
    for i, (name, lat, lon) in enumerate(sites):
        if inside_mask is not None:
            inside = bool(inside_mask[i])
        else:
            inside = ((lon - mean_long) / (n_std * std_long)) ** 2 + \
                     ((lat - mean_lat) / (n_std * std_lat)) ** 2 <= 1.0
        if inside:
            dist_km = config.KM_PER_DEGREE * np.sqrt(
                ((lon - mean_long) * cos_ref) ** 2 + (lat - mean_lat) ** 2
            )
            rows.append({'site': name, 'lat': lat, 'lon': lon, 'dist_to_mean_km': dist_km})
    columns = ['site', 'lat', 'lon', 'dist_to_mean_km']
    return pd.DataFrame(rows, columns=columns).sort_values('dist_to_mean_km').reset_index(drop=True)


def minerals_within_ellipse(mean_long, mean_lat, std_long, std_lat, n_std=2.0, deposits=None, cov=None):
    """Which known mineral deposits (load_mineral_deposits(), or a
    caller-supplied `deposits` list of (locality, metal, lat, lon)) fall
    geometrically inside this n_std confidence ellipse. Same city-agnostic
    pooling as sites_within_ellipse(); same optional `cov` argument too --
    see that function's docstring for why (uses the true, possibly-rotated
    ellipse when given, the axis-aligned approximation otherwise).
    Informational only -- see this module's docstring for why mineral
    proximity is never used to score, rank, or exclude a region.

    Returns a DataFrame (locality, metal, lat, lon, dist_to_mean_km),
    empty if none fall inside."""
    deposits = deposits if deposits is not None else load_mineral_deposits()
    cos_ref = np.cos(np.radians(config.LATITUDE_PARAM))
    if cov is not None:
        dep_lons = np.array([d[3] for d in deposits])
        dep_lats = np.array([d[2] for d in deposits])
        inside_mask = _mahalanobis_mask(dep_lons, dep_lats, mean_long, mean_lat, cov, n_std)
    else:
        inside_mask = None
    rows = []
    for i, (locality, metal, lat, lon) in enumerate(deposits):
        if inside_mask is not None:
            inside = bool(inside_mask[i])
        else:
            inside = ((lon - mean_long) / (n_std * std_long)) ** 2 + \
                     ((lat - mean_lat) / (n_std * std_lat)) ** 2 <= 1.0
        if inside:
            dist_km = config.KM_PER_DEGREE * np.sqrt(
                ((lon - mean_long) * cos_ref) ** 2 + (lat - mean_lat) ** 2
            )
            rows.append({'locality': locality, 'metal': metal, 'lat': lat, 'lon': lon,
                         'dist_to_mean_km': dist_km})
    columns = ['locality', 'metal', 'lat', 'lon', 'dist_to_mean_km']
    return pd.DataFrame(rows, columns=columns).sort_values('dist_to_mean_km').reset_index(drop=True)


def geo_screened_region(pts, n_std=2.0, grid_res_deg=None, max_slope_deg=15.0,
                         river_buffer_km=0.2, water=None):
    """A refined candidate region, replacing invented per-criterion
    weights with hard geographic exclusions -- the approach discussed
    with the user as an alternative to fabricating coefficients.

    Starts from the TRUE confidence ellipse of the bootstrap sample cloud
    `pts` (an (N,2) array of [lon, lat] draws) -- using the full
    covariance matrix via Mahalanobis distance, not the axis-aligned
    approximation sites_within_ellipse()/screen_ellipse() use, so
    correlation between predicted longitude and latitude is respected.
    Then carves out grid cells that fail hard, independently-justifiable
    exclusions:
      - max_slope_deg (default 15 degrees): a commonly-cited practical
        threshold for pre-modern settlement siting. This is a modeling
        choice, not a value derived from this dataset -- adjust freely.
        NOTE this is the one exclusion NOT sharpened by a finer
        grid_res_deg: slope is a finite difference on the FAO-GAEZ
        elevation grid, which is natively ~9km resolution, so nearby
        fine-grid points sampling the same coarse DEM cell get nearly
        identical slope values regardless of how densely you sample. A
        finer grid makes the water/known-site exclusions and the
        polygon's visual boundary sharper; it does not invent elevation
        detail the source DEM doesn't have. If you want a visibly
        smaller/more selective region, max_slope_deg or n_std are the
        real levers, not grid_res_deg.
      - river_buffer_km (default 0.2 km) / lake polygons: excludes only
        cells essentially ON a river channel or inside a lake, not the
        surrounding floodplain -- being NEAR water is desirable for a
        settlement, only being IN it is disqualifying, so this is a tight
        buffer, not a proximity preference.
    grid_res_deg default is ~1km (was the DEM's native ~9km cell size --
    too coarse to meaningfully ask "is there a specific ~1km spot here
    suitable," which is the actual question for siting one settlement).
    Water/known-site exclusions are genuinely sharper at 1km; see the
    max_slope_deg note above for why slope itself isn't.

    The result is frequently non-convex or split into disconnected
    pieces -- that's expected, not a bug: a river or ridge cutting
    through a confidence ellipse should split it, an ellipse can't
    represent that.

    Returns (polygon, grid_df): polygon is a shapely Polygon/MultiPolygon
    (None if nothing in the ellipse passes every exclusion), grid_df has
    every evaluated grid cell with slope and pass/fail, for transparency.
    """
    pts = np.asarray(pts, dtype=float)
    mean = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    std_lon, std_lat = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

    grid_res_deg = grid_res_deg or (1.0 / config.KM_PER_DEGREE)  # ~1km

    lon_min, lon_max = mean[0] - n_std * std_lon, mean[0] + n_std * std_lon
    lat_min, lat_max = mean[1] - n_std * std_lat, mean[1] + n_std * std_lat
    lon_vals = np.arange(lon_min, lon_max + grid_res_deg, grid_res_deg)
    lat_vals = np.arange(lat_min, lat_max + grid_res_deg, grid_res_deg)
    lon_mesh, lat_mesh = np.meshgrid(lon_vals, lat_vals)  # row i -> lat_vals[i] (south to north)
    lon_flat, lat_flat = lon_mesh.ravel(), lat_mesh.ravel()

    d = np.column_stack([lon_flat - mean[0], lat_flat - mean[1]])
    mahal2 = np.einsum('ij,jk,ik->i', d, cov_inv, d)
    in_ellipse = mahal2 <= n_std ** 2

    slope = np.full(len(lon_flat), np.nan)
    if in_ellipse.any():
        slope[in_ellipse] = sample_slope(lon_flat[in_ellipse], lat_flat[in_ellipse])
    not_too_steep = np.where(in_ellipse, slope <= max_slope_deg, False)

    water = water if water is not None else load_water_geometries()
    buffer_deg = river_buffer_km / config.KM_PER_DEGREE
    # Buffer once, then use a prepared geometry for fast repeated .contains()
    # checks (spatial-indexed) instead of computing .distance() to every
    # water feature for every point -- needed now that grid_res_deg~1km
    # means far more points than the old ~9km grid did.
    exclusion_geom = water.union_all().buffer(buffer_deg) if hasattr(water, 'union_all') \
        else water.unary_union.buffer(buffer_deg)
    exclusion_prepared = prepared_geom(exclusion_geom)
    not_in_water = np.ones(len(lon_flat), dtype=bool)
    for i in np.where(in_ellipse & not_too_steep)[0]:
        if exclusion_prepared.contains(Point(lon_flat[i], lat_flat[i])):
            not_in_water[i] = False

    passable = in_ellipse & not_too_steep & not_in_water
    grid_df = pd.DataFrame({
        'lon': lon_flat, 'lat': lat_flat, 'in_ellipse': in_ellipse,
        'slope_deg': slope, 'passable': passable,
    })

    polygon = None
    if passable.any():
        mask2d = passable.reshape(lon_mesh.shape)
        mask2d_northup = np.flipud(mask2d).astype(np.uint8)  # raster convention: row 0 = north
        transform = from_origin(
            lon_vals[0] - grid_res_deg / 2, lat_vals[-1] + grid_res_deg / 2, grid_res_deg, grid_res_deg,
        )
        polys = [shapely_shape(geom) for geom, val in
                 rio_shapes(mask2d_northup, mask=mask2d_northup.astype(bool), transform=transform)
                 if val == 1]
        if polys:
            polygon = unary_union(polys)

    return polygon, grid_df


def sites_within_polygon(polygon, sites=None):
    """Which known archaeological sites fall inside a (possibly
    non-convex or multi-part) region, e.g. from geo_screened_region().
    City-agnostic, same principle as sites_within_ellipse()."""
    sites = sites if sites is not None else data.KNOWN_ARCHAEOLOGICAL_SITES
    columns = ['site', 'lat', 'lon']
    if polygon is None:
        return pd.DataFrame(columns=columns)
    rows = [{'site': name, 'lat': lat, 'lon': lon}
            for name, lat, lon in sites if polygon.contains(Point(lon, lat))]
    return pd.DataFrame(rows, columns=columns).sort_values('site').reset_index(drop=True)


def load_mineral_deposits():
    """66 Early-Bronze-Age-era mineral deposits (copper/silver/gold/tin)
    from Barjamovic et al.'s own replication package
    (config.MINERAL_DEPOSITS_DTA), each as (locality, metal, lat, lon).
    Cached module-level, same pattern as the water loader.
    Informational only -- see this module's docstring for why it is never
    used to score or exclude a region."""
    global _MINERALS_CACHE
    if _MINERALS_CACHE is None:
        df = pd.read_stata(config.MINERAL_DEPOSITS_DTA)
        _MINERALS_CACHE = [
            (row['locality'], row['metal'], float(row['lat_y']), float(row['long_x']))
            for _, row in df.iterrows()
        ]
    return _MINERALS_CACHE


def minerals_within_polygon(polygon, deposits=None):
    """Which known mineral deposits fall inside a (possibly non-convex or
    multi-part) region, e.g. from geo_screened_region(). Same
    city-agnostic containment check as sites_within_polygon() -- shown on
    plots/tables for context, never used to include, exclude, or rank any
    candidate area."""
    deposits = deposits if deposits is not None else load_mineral_deposits()
    columns = ['locality', 'metal', 'lat', 'lon']
    if polygon is None:
        return pd.DataFrame(columns=columns)
    rows = [{'locality': name, 'metal': metal, 'lat': lat, 'lon': lon}
            for name, metal, lat, lon in deposits if polygon.contains(Point(lon, lat))]
    return pd.DataFrame(rows, columns=columns).sort_values('locality').reset_index(drop=True)


def sample_ellipse_grid(mean_long, mean_lat, std_long, std_lat, n_std=2.0, n_per_axis=15, cov=None):
    """Uniform candidate grid inside the n_std confidence ellipse around
    (mean_long, mean_lat).

    cov: optional 2x2 covariance matrix of the bootstrap prediction cloud
    (e.g. built from a results row's Var_Long/Var_Lat/Cov_LonLat -- see
    evaluate.summarize_bootstrap()). When given, this samples the TRUE,
    possibly-rotated ellipse (same Mahalanobis-distance definition as
    plots.draw_confidence_ellipse() and geo_screened_region() already
    use), so a city whose longitude and latitude errors are correlated
    across bootstrap runs gets a correctly tilted candidate region instead
    of an axis-aligned one. When cov is None (e.g. no covariance was
    saved for this results file), falls back to the axis-aligned
    mean+/-std approximation this function used before -- a superset of
    the true ellipse (over-covers rather than misses candidates), fine as
    a fallback but not exact.

    Fixed point count per axis (not a fixed physical resolution like
    geo_screened_region()'s grid_res_deg): tried the fixed-~1km-resolution
    version, but the resulting grid (much denser, and blocky/square at
    the ellipse boundary at this point spacing) looked worse in practice
    than this coarser, fixed-count version -- reverted."""
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        # The axis-aligned bounding box of ANY Mahalanobis ellipse
        # (x-mean)^T Sigma^-1 (x-mean) <= n_std^2 has half-width
        # n_std*sqrt(Sigma[i,i]) along dimension i -- exact, not an
        # approximation, and needs no eigendecomposition: the ellipse's
        # extreme point along axis i, however it's rotated, always
        # reaches exactly that far (a standard property of confidence
        # ellipses). We only sample a rectangular grid over this box here;
        # _mahalanobis_mask() below (not this box) is what actually
        # decides which of those grid points lie inside the true, rotated
        # ellipse.
        half_lon = n_std * np.sqrt(max(cov[0, 0], 1e-12))
        half_lat = n_std * np.sqrt(max(cov[1, 1], 1e-12))
        lon_grid = np.linspace(mean_long - half_lon, mean_long + half_lon, n_per_axis)
        lat_grid = np.linspace(mean_lat - half_lat, mean_lat + half_lat, n_per_axis)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        lon_flat, lat_flat = lon_mesh.ravel(), lat_mesh.ravel()
        inside = _mahalanobis_mask(lon_flat, lat_flat, mean_long, mean_lat, cov, n_std)
        return lon_flat[inside], lat_flat[inside]

    lon_grid = np.linspace(mean_long - n_std * std_long, mean_long + n_std * std_long, n_per_axis)
    lat_grid = np.linspace(mean_lat - n_std * std_lat, mean_lat + n_std * std_lat, n_per_axis)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    lon_flat, lat_flat = lon_mesh.ravel(), lat_mesh.ravel()

    inside = ((lon_flat - mean_long) / (n_std * std_long)) ** 2 + \
             ((lat_flat - mean_lat) / (n_std * std_lat)) ** 2 <= 1.0
    return lon_flat[inside], lat_flat[inside]


def screen_ellipse(mean_long, mean_lat, std_long, std_lat, n_std=2.0, n_per_axis=15,
                    max_slope_deg=None, water=None, sites=None, include_known_sites=False, cov=None):
    """Ranks candidate cells inside a lost city's confidence ellipse by
    slope/elevation, distance to water, distance to the nearest KNOWN
    archaeological site (data.KNOWN_ARCHAEOLOGICAL_SITES, or a
    caller-supplied `sites` list) -- whichever site that happens to be,
    not one pre-assigned to this particular lost city -- and distance to
    the model's own point prediction (mean_long, mean_lat, i.e. the
    ellipse's center). That last criterion needs no separate lookup: the
    ellipse is already centered on the reported prediction for whichever
    results file the caller loaded (results/ancient_lost_dyadic_
    constrained.parquet by default -- see experiments/
    screening_ancient_lost.py's RESULTS_NAME), so "close to the ellipse
    center" and "close to the constrained prediction" are the same thing
    by construction. Best candidates first.

    Mineral deposits are deliberately NOT part of this ranking (see this
    module's docstring) -- they're a resource point, not a candidate
    settlement location, and Barjamovic et al.'s own analysis found
    distance to them not predictive. They're shown as informational
    context elsewhere (minerals_within_polygon(), used by
    ancient_lost_dyadic(_constrained).py), never scored or ranked here.

    include_known_sites: if True, any known archaeological site that
    falls inside this ellipse is added to the candidate pool as an extra
    row (marked in the 'known_site' column) and ranked on the exact same
    criteria as every grid cell -- this is what answers "does a real,
    named site happen to sit in a good spot," rather than only telling
    you the best *unnamed* grid coordinates. A known site's own
    dist_known_site_km is computed against every OTHER known site
    (excluding itself), not the full pool -- otherwise it would trivially
    score a perfect zero for being a known site next to itself.

    No single combination formula for the remaining criteria is specified
    in HANDOFF.md §7.3 (unlike §7.1's constraints, which had an exact
    formula) -- this uses a simple rank-sum, easy to replace with a
    different weighting.

    cov: optional 2x2 covariance matrix of the bootstrap prediction cloud
    (see sample_ellipse_grid()'s docstring). When given, both the
    candidate grid and the known-site containment check use the TRUE,
    possibly-rotated ellipse instead of the axis-aligned mean+/-std
    approximation, so this matches the ellipse actually drawn on the
    per-city uncertainty plot (plots.draw_confidence_ellipse()) and the
    geography-carved region (geo_screened_region()) for the same city."""
    lons, lats = sample_ellipse_grid(mean_long, mean_lat, std_long, std_lat, n_std, n_per_axis, cov=cov)
    known_site_names = [None] * len(lons)

    if include_known_sites:
        site_pool = sites if sites is not None else data.KNOWN_ARCHAEOLOGICAL_SITES
        in_ellipse = sites_within_ellipse(mean_long, mean_lat, std_long, std_lat, n_std,
                                           sites=site_pool, cov=cov)
        if len(in_ellipse) > 0:
            lons = np.concatenate([lons, in_ellipse['lon'].values])
            lats = np.concatenate([lats, in_ellipse['lat'].values])
            known_site_names = known_site_names + list(in_ellipse['site'])

    columns = ['lat', 'lon', 'known_site', 'elevation_m', 'slope_deg', 'dist_water_km',
               'dist_known_site_km', 'dist_to_prediction_km', 'in_lake', 'rank_score']
    if len(lons) == 0:
        return pd.DataFrame(columns=columns)

    site_pool = sites if sites is not None else data.KNOWN_ARCHAEOLOGICAL_SITES
    dist_known_site_km = hoyuk_proximity_score(lons, lats, sites=site_pool)
    for i, site_name in enumerate(known_site_names):
        if site_name is not None:
            # Exclude the site itself so it isn't scored as "0km from a
            # known site" purely for being one.
            other_sites = [s for s in site_pool if s[0] != site_name]
            dist_known_site_km[i] = hoyuk_proximity_score([lons[i]], [lats[i]], sites=other_sites)[0]

    cos_ref = np.cos(np.radians(config.LATITUDE_PARAM))
    dist_to_prediction_km = config.KM_PER_DEGREE * np.sqrt(
        ((np.asarray(lons) - mean_long) * cos_ref) ** 2 + (np.asarray(lats) - mean_lat) ** 2
    )

    df = pd.DataFrame({
        'lat': lats, 'lon': lons, 'known_site': known_site_names,
        'elevation_m': sample_elevation(lons, lats),
        'slope_deg': sample_slope(lons, lats),
        'dist_water_km': distance_to_water_km(lons, lats, water=water),
        'dist_known_site_km': dist_known_site_km,
        'dist_to_prediction_km': dist_to_prediction_km,
        'in_lake': in_lake_mask(lons, lats),
    })

    if max_slope_deg is not None:
        df = df[df['slope_deg'] <= max_slope_deg].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=columns)

    df['rank_score'] = (
        df['dist_water_km'].rank(ascending=True)
        + df['slope_deg'].rank(ascending=True)
        + df['dist_known_site_km'].rank(ascending=True)
        + df['dist_to_prediction_km'].rank(ascending=True)
    )
    # dist_water_km above rewards proximity to ANY water, rivers included --
    # reasonable in general (irrigation/drinking access), but wrong for a
    # cell that sits ON a lake itself (e.g. Tuz Golu): that's not "close to
    # water," it's water. Push those rows to worse than every ordinarily-
    # ranked candidate, regardless of how favorable their other three
    # criteria look, rather than leaving them to be rewarded by
    # dist_water_km's raw proximity score. Rivers are deliberately exempt
    # (see in_lake_mask()'s docstring) -- only lakes get this hard penalty.
    if df['in_lake'].any():
        df.loc[df['in_lake'], 'rank_score'] = df['rank_score'].max() + 1
    return df.sort_values('rank_score').reset_index(drop=True)


def screen_all_lost_cities(summary_df, n_std=2.0, n_per_axis=15, max_slope_deg=None,
                            include_known_sites=False):
    """Runs screen_ellipse() for every city in a results summary DataFrame
    (e.g. from evaluate.run_ancient_lost_prediction()). Returns
    {city_name: ranked DataFrame}, loading the water/site layers once and
    reusing them across cities. The known-site criterion is the same
    city-agnostic pool for every city -- see screen_ellipse().

    If `summary_df` has Var_Long/Var_Lat/Cov_LonLat columns (saved by
    evaluate.summarize_bootstrap() for any results file produced after
    this covariance fix), each city's true, possibly-rotated ellipse is
    used automatically -- matching the ellipse drawn on that city's own
    uncertainty plot, instead of an axis-aligned mean+/-std approximation
    that ignores correlation between a city's longitude and latitude
    errors. Older results files without those columns still work exactly
    as before (screen_ellipse()'s cov=None fallback)."""
    water = load_water_geometries()
    has_cov = {'Var_Long', 'Var_Lat', 'Cov_LonLat'}.issubset(summary_df.columns)
    results = {}
    for _, row in summary_df.iterrows():
        cov = None
        if has_cov:
            cov = np.array([[row['Var_Long'], row['Cov_LonLat']],
                             [row['Cov_LonLat'], row['Var_Lat']]])
        results[row['City']] = screen_ellipse(
            row['Mean_Long'], row['Mean_Lat'], row['Std_Long'], row['Std_Lat'],
            n_std=n_std, n_per_axis=n_per_axis, max_slope_deg=max_slope_deg,
            water=water, include_known_sites=include_known_sites, cov=cov,
        )
    return results
