# -*- coding: utf-8 -*-
"""
Shared map / confidence-ellipse plotting.

Bug 6 fix (cosmetic, but the repo goes public): the original
GCNDyadicBootstrap.py hardcoded "TII" into its aggregate and
per-city plot titles even though it is the Dyadic-share script.
Every title here is built from a `model_label` argument the caller
passes explicitly, so a Dyadic run can never be mislabeled TII again.

Figures are always saved to disk (results/figures/). Whether they ALSO
pop up on screen depends on how the script is run:
  - Run from Spyder (or any interactive session): they show normally,
    exactly like the legacy scripts' plt.show() calls, because Spyder
    manages its own matplotlib backend.
  - Run headlessly (e.g. Claude driving a background PowerShell/Bash
    process for a long bootstrap): set the environment variable
    GCN_HEADLESS=1 first. That forces the non-interactive 'Agg' backend
    so a batch of 200 runs x 25 cities can't hang waiting for a window
    to be closed. Without it, the default backend's plt.show() would
    block on every figure.

Basemap failures are now printed, never silently swallowed. If you see
"basemap fetch failed" in the console, the physical/topographic map is
missing from that figure -- check your internet connection. This was
previously a bare try/except that hid the failure entirely.

FIGSIZE/DPI are fixed module-level constants used by every figure type
(individual uncertainty plots and the aggregate map), so every image
this module produces has identical pixel dimensions for the article.
"""

import os

import matplotlib
if os.environ.get("GCN_HEADLESS"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon as MplPolygon
import matplotlib.patheffects as path_effects
import numpy as np

# Thin black halo behind label text -- otherwise light-colored labels
# (e.g. the orange known-site / indigo mineral names) disappear against
# the viridis_r gradient or basemap wherever they happen to land on a
# similarly light patch. Reused by every annotate() call that labels a
# point on a busy/colored background.
_TEXT_HALO = [path_effects.withStroke(linewidth=2.5, foreground='black')]

from . import config

try:
    import geopandas as gpd
    import contextily as cx
    _HAS_MAP_LIBS = True
except ImportError:
    _HAS_MAP_LIBS = False

FIGSIZE = (12, 10)
DPI = 150

def _model_family(model_label):
    """Splits a model_label like "MDS-Dyadic" into ("MDS", "Dyadic"). Every
    existing GCN experiment script passes a bare "Dyadic"/"TII" (no "-"),
    which is why bare labels default to family "GCN" here -- this keeps
    every existing GCN filename/legend-label byte-for-byte unchanged.
    Previously this module hardcoded model_type="GCN" outright, with a
    comment claiming "this repository only ever produces GCN figures" --
    false as soon as MDS reuses this module too, and exactly the kind of
    hardcoded-label mistake this module's own Bug 6 fix (see top of file)
    was supposed to make impossible. mds_*.py experiment scripts pass
    model_label="MDS-Dyadic"/"MDS-TII" specifically so this split works."""
    base = model_label.split(" (")[0]
    if "-" in base:
        family, _, weighting = base.partition("-")
        return family, weighting
    return "GCN", base


def _build_filename(model_label, category, name=None):
    """"GCN - Dyadic/TII - LOO/Modern/Lost[ (Constrained)] - <City>.png"
    (or "MDS - Dyadic/TII - ..." for MDS runs), or without the trailing
    city segment for an aggregate/combined map. model_label may carry a
    parenthetical qualifier (e.g. "Dyadic (constrained)") for the plot
    title; only the weighting itself (Dyadic/TII) belongs in the filename,
    since `category` already carries any constrained/unconstrained
    distinction."""
    family, weighting = _model_family(model_label)
    parts = [family, weighting, category]
    if name:
        parts.append(name)
    else:
        parts.append("Aggregate")
    return " - ".join(parts) + ".png"


def _iter_rings(geom):
    """Yields every LinearRing (exterior + holes) of a Polygon or
    MultiPolygon, flattening MultiPolygon parts."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        yield geom.exterior
        yield from geom.interiors
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:
            yield from _iter_rings(part)


def _plot_polygon_outlines(ax, geometries, **kwargs):
    """Draws each polygon ring as a plain Line2D via ax.plot(), instead
    of geopandas.GeoDataFrame.plot()'s PathCollection-based renderer.

    geopandas's own .plot() crashes natively here (Fatal Python error:
    Aborted, inside matplotlib's PathCollection.get_datalim ->
    get_affine) when this module is run inside Spyder's interactive Qt
    backend -- reproducible on the very first plot call in a fresh
    kernel, every time, but never reproducible running the identical
    code/environment as a plain headless script (same root cause class
    as the rasterio sample_elevation() and geopandas distance_to_water_km()
    native crashes fixed elsewhere in this module/terrain.py -- avoid the
    fragile C-extension-heavy call rather than debug the C extension
    itself). Since this call only ever draws unfilled borders
    (color='none'), a ring-by-ring line plot is visually identical to
    the PathCollection version and never touches that code path."""
    for geom in geometries:
        for ring in _iter_rings(geom):
            xs, ys = ring.xy
            ax.plot(xs, ys, **kwargs)


def load_and_plot_turkey_map(ax, shapefile_path=None):
    shapefile_path = shapefile_path or config.SHAPEFILE_PATH
    if not _HAS_MAP_LIBS:
        print("plots.py: geopandas/contextily not installed -- no borders or basemap drawn.")
        return
    if os.path.exists(shapefile_path):
        try:
            turkey_map = gpd.read_file(shapefile_path).to_crs(epsg=4326)
            _plot_polygon_outlines(ax, turkey_map.geometry, color='black', linewidth=0.8, alpha=0.6, zorder=2)
        except Exception as e:
            print(f"plots.py: Turkey border shapefile failed to load/plot ({e}) -- borders missing from this figure.")
    else:
        print(f"plots.py: shapefile not found at {shapefile_path} -- borders missing from this figure.")
    try:
        # attribution=False: contextily's add_basemap() only forces an
        # early, partial fig.canvas.draw() (before every other artist on
        # this Axes is added) to measure/position its own "(C) OpenStreetMap
        # contributors" watermark text -- see add_attribution() in
        # contextily/plotting.py. That premature draw is what aborts here
        # (Fatal Python error: Aborted, inside matplotlib's transform
        # code) -- a known contextily/matplotlib interaction issue, not
        # something wrong with the tiles or our own plotted data. Turning
        # off the watermark skips that forced draw entirely; the basemap
        # tiles themselves are unaffected. Since the watermark won't be on
        # the figure, credit OpenTopoMap/OpenStreetMap in the manuscript's
        # figure caption or methods text instead.
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenTopoMap, zorder=0, alpha=0.6,
                        attribution=False)
    except Exception as e:
        print(f"plots.py: basemap fetch failed ({type(e).__name__}: {e}) -- "
              f"physical/topographic map missing from this figure. Usually a network/SSL issue "
              f"reaching the tile server; check your internet connection.")


def _ellipse_patch_from_cov(mean, cov, n_std, facecolor='none', **kwargs):
    """The single definition of "the confidence ellipse" as an actual
    matplotlib Ellipse patch (a smooth curve, not a discretized/sampled
    approximation of one) -- shared by draw_confidence_ellipse() below
    (which computes mean/cov from raw bootstrap points) and
    plot_screening_ranking()'s ellipse overlay (which reuses this exact
    same shape from a results row's saved Var_Long/Var_Lat/Cov_LonLat, so
    the screening plot's boundary is never a cruder approximation of the
    uncertainty plot's own ellipse for the same city)."""
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))
    return Ellipse(xy=mean, width=width, height=height, angle=theta, facecolor=facecolor, **kwargs)


def draw_confidence_ellipse(data, ax, n_std=1.0, facecolor='none', **kwargs):
    if len(data) < 3:
        return None
    cov = np.cov(data, rowvar=False)
    mean = np.mean(data, axis=0)
    return ax.add_patch(_ellipse_patch_from_cov(mean, cov, n_std, facecolor=facecolor, **kwargs))


def draw_geo_screened_region(polygon, ax, **kwargs):
    """Draws a shapely Polygon or MultiPolygon (e.g. from
    terrain.geo_screened_region()) -- the geography-carved region, which
    is frequently non-convex or multi-part, unlike the elliptical
    statistical region. Plots.py doesn't know or care why the shape is
    the way it is; that logic lives entirely in terrain.py."""
    if polygon is None:
        return []
    geoms = list(polygon.geoms) if hasattr(polygon, 'geoms') else [polygon]
    patches = []
    for i, geom in enumerate(geoms):
        outer_kwargs = dict(kwargs)
        if i > 0:
            outer_kwargs.pop('label', None)  # only the first sub-polygon gets a legend entry
        patches.append(ax.add_patch(MplPolygon(list(geom.exterior.coords), closed=True, **outer_kwargs)))
        for interior in geom.interiors:
            hole_kwargs = dict(kwargs)
            hole_kwargs['facecolor'] = 'none'
            hole_kwargs.pop('label', None)
            patches.append(ax.add_patch(MplPolygon(list(interior.coords), closed=True, **hole_kwargs)))
    return patches


def plot_individual_uncertainty(name, pts, model_label, n_runs, category, actual=None, comparison=None,
                                 comparison_label='Barjamovic et al. (gravity model)',
                                 barjamovic_2011=None, forlanini_2008=None,
                                 known_sites=None, minerals=None, geo_screened_polygon=None,
                                 save_dir=None, shapefile_path=None):
    """category: one of "LOO", "Modern", "Lost", or "Lost (Constrained)"
    -- which experiment this plot belongs to, used only to build the
    saved filename (see _build_filename() above), not shown on the plot
    itself (the title already states this via model_label/n_runs).
    known_sites: optional DataFrame with columns (site, lat, lon, ...),
    e.g. from terrain.sites_within_ellipse() or terrain.sites_within_polygon()
    -- plotted as triangle markers with name labels.
    minerals: optional DataFrame with columns (locality, metal, lat, lon),
    e.g. from terrain.minerals_within_polygon() -- plotted as diamond
    markers with "locality (metal)" labels. Informational only: shown for
    context, never used to include/exclude or score any region -- see
    terrain.py's module docstring for why.
    geo_screened_polygon: optional shapely Polygon/MultiPolygon, e.g. from
    terrain.geo_screened_region() -- the geography-carved, non-elliptical
    candidate region, drawn as a filled purple outline over the
    statistical ellipses.
    barjamovic_2011/forlanini_2008: optional (lon, lat) tuples -- the
    historians' own philological/archaeological proposals (Barjamovic's
    2011 monograph, Forlanini 2008), distinct from `comparison`
    (Barjamovic et al.'s later, separate structural GRAVITY MODEL point
    estimate from the QJE paper -- a different kind of estimate from the
    same author). Drawn as 'B'/'F' letter markers, matching the style of
    HistoriansAndKnownCities.py. Intentionally only available on this
    per-city plot, not plot_aggregate_map() below -- too cluttered there.
    Left to the caller to compute (plots.py stays geography-agnostic;
    terrain.py/data.py decide what these values are)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    load_and_plot_turkey_map(ax, shapefile_path)

    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=15, color='gray', label='MC Samples')
    draw_confidence_ellipse(pts, ax, n_std=1.0, edgecolor='blue', label='1-Sigma (68%)')
    draw_confidence_ellipse(pts, ax, n_std=2.0, edgecolor='red', label='2-Sigma (95%)')
    if geo_screened_polygon is not None:
        draw_geo_screened_region(geo_screened_polygon, ax, edgecolor='purple', facecolor='purple',
                                  alpha=0.15, linewidth=1.5, linestyle='--', zorder=3,
                                  label='Geo-screened region')
    mean_label = f"{_model_family(model_label)[0]} Mean"
    ax.scatter(pts[:, 0].mean(), pts[:, 1].mean(), c='black', marker='x', s=100, label=mean_label, zorder=5)

    # Zoom includes every plotted point, however far -- if the GCN and a
    # historian disagree by hundreds of km, that disagreement should be
    # visible on the map, not clipped out of frame.
    xs, ys = [pts[:, 0].min(), pts[:, 0].max()], [pts[:, 1].min(), pts[:, 1].max()]

    def _plot_comparison(point, **scatter_kwargs):
        if point is None:
            return
        ax.scatter(point[0], point[1], zorder=scatter_kwargs.pop('zorder', 5), **scatter_kwargs)
        xs.append(point[0]); ys.append(point[1])

    _plot_comparison(actual, c='blue', marker='o', s=120, edgecolors='k', label='Actual', zorder=5)
    _plot_comparison(comparison, c='green', marker='s', s=80, edgecolors='k', label=comparison_label, zorder=4)
    _plot_comparison(barjamovic_2011, marker='$B$', s=170, color='darkgreen', linewidth=0.4,
                      label='Barjamovic 2011 (historian)', zorder=6)
    _plot_comparison(forlanini_2008, marker='$F$', s=170, color='darkred', linewidth=0.4,
                      label='Forlanini 2008 (historian)', zorder=6)

    if known_sites is not None and len(known_sites) > 0:
        ax.scatter(known_sites['lon'], known_sites['lat'], c='orange', marker='^', s=110,
                   edgecolors='k', zorder=6, label='Known Site')
        xs += known_sites['lon'].tolist(); ys += known_sites['lat'].tolist()
        for _, site_row in known_sites.iterrows():
            ax.annotate(site_row['site'], (site_row['lon'], site_row['lat']),
                        fontsize=7, color='darkorange', xytext=(4, 4), textcoords='offset points')

    if minerals is not None and len(minerals) > 0:
        ax.scatter(minerals['lon'], minerals['lat'], c='mediumpurple', marker='D', s=70,
                   edgecolors='k', zorder=6, label='Mineral Deposit')
        xs += minerals['lon'].tolist(); ys += minerals['lat'].tolist()
        for _, ore_row in minerals.iterrows():
            ax.annotate(f"{ore_row['locality']} ({ore_row['metal']})", (ore_row['lon'], ore_row['lat']),
                        fontsize=6, color='indigo', xytext=(4, -8), textcoords='offset points')

    pad = 0.7
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_title(f"Uncertainty Quantification ({model_label}): {name}\n(N={n_runs} Bootstrap Runs)")
    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, _build_filename(model_label, category, name)),
                    dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)
    return fig


def plot_screening_ranking(name, screened_df, category, model_label='Dyadic', save_dir=None,
                            shapefile_path=None, comparison=None,
                            comparison_label='Barjamovic et al. (gravity model)',
                            barjamovic_2011=None, forlanini_2008=None,
                            gcn_mean=None, minerals=None, ellipse_cov=None, ellipse_n_std=2.0):
    """screened_df: output of terrain.screen_ellipse()/screen_all_lost_cities(),
    one row per candidate grid cell inside a lost city's confidence
    ellipse, plus (if that call used include_known_sites=True) one row
    per known archaeological site that also falls inside the ellipse,
    flagged in the 'known_site' column.

    Shows rank_score as a continuous filled-contour gradient across the
    grid cells (terrain.screen_ellipse()'s docstring has the exact
    rank-sum formula), interpolated from the scored grid points via
    ax.tricontourf() -- not a per-cell scatter, which at this point
    density read as a wall of disconnected dots rather than a legible
    spatial pattern. Does NOT mark individual top-N candidates on the
    map (that ranking is reported as its own table -- see
    experiments/screening_ancient_lost.py's console/CSV output -- circles
    and rank numbers on top of the gradient were unreadable clutter, not
    informative). Known archaeological sites are drawn as plain
    orange triangles (matching plot_individual_uncertainty's convention)
    -- deliberately NOT colored by rank_score: mixing a categorical
    "this is a real, named site" marker into the same continuous
    scientific color scale as the interpolated surface was confusing,
    not informative (their rank is already in the console report / the
    dist_to_prediction_km column, not something this map needs to encode
    a second time visually).

    comparison/barjamovic_2011/forlanini_2008/gcn_mean: optional (lon,
    lat) points, same convention as plot_individual_uncertainty -- always
    included in the view regardless of distance, so this figure and the
    per-city uncertainty plot show a consistent comparison set. gcn_mean
    is this run's own bootstrap-mean point prediction (the same point
    plot_individual_uncertainty marks as "GCN Mean") -- i.e. the ellipse's
    own center, explicitly marked here too since it's otherwise implicit
    in the contour surface, not visibly called out.

    minerals: optional DataFrame (locality, metal, lat, lon, ...), e.g.
    from terrain.minerals_within_ellipse() -- shown as purple diamonds,
    same style/color as plot_individual_uncertainty's mineral markers and
    deliberately NOT on the rank_score color scale (a different color
    from the orange known-site triangles, per the same "don't mix a
    categorical marker into the continuous scale" reasoning). Shown for
    context only -- see this module's and terrain.py's docstrings for why
    mineral proximity was never part of rank_score.

    ellipse_cov: optional 2x2 covariance matrix (a results row's
    Var_Long/Var_Lat/Cov_LonLat -- see terrain.screen_ellipse()'s
    docstring). When given (together with gcn_mean as the ellipse
    center), the exact same smooth ellipse drawn on this city's
    plot_individual_uncertainty figure is (a) drawn here too as a crisp
    outline, and (b) used to clip the interpolated rank_score surface to
    that precise boundary. screen_ellipse()'s candidate grid is a sparse,
    discretized approximation of this same ellipse (coarse enough that
    tricontourf's triangulation of it can look like a jagged polygon,
    especially once the ellipse is rotated) -- the grid still determines
    what gets scored and ranked, but the plotted boundary is the true
    ellipse, not the grid's own jagged edge."""
    if screened_df is None or len(screened_df) == 0:
        print(f"plots.py: no candidate cells to plot for {name} (empty screened_df) -- skipping.")
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    load_and_plot_turkey_map(ax, shapefile_path)

    # Bug fix (found via the MDS screening script's wider ellipses -- see
    # CHANGELOG_AND_HANDOFF.md's MDS section, Bug M7): a cell/site near a
    # raster edge (e.g. a coastline) can have a NaN rank_score if
    # terrain.sample_slope()'s neighbor-point sampling falls outside the
    # elevation raster's coverage. tricontourf() below raises on any NaN
    # z-value, so those cells are excluded from the contour surface --
    # they were never meaningfully rankable (missing terrain data), so
    # dropping them from the plotted surface is correct, not just a crash
    # workaround. Unscored known sites are still listed in the console
    # output (see experiments/screening_ancient_lost.py) with an explicit
    # "rank unavailable" note; only the map surface skips them here.
    grid_df = screened_df[screened_df['known_site'].isna() & screened_df['rank_score'].notna()]
    site_df = screened_df[screened_df['known_site'].notna()]

    # Without an explicit zoom, the view defaults to the union of every
    # plotted artist -- including the full-country border/basemap from
    # load_and_plot_turkey_map() above -- so the ellipse's small cluster
    # of candidate cells would be lost in the corner of a whole-Turkey
    # map. Zoom to the candidate region, but widen to also include every
    # comparison point regardless of how far it is -- same "however far"
    # principle as plot_individual_uncertainty's xs/ys zoom.
    xs = [screened_df['lon'].min(), screened_df['lon'].max()]
    ys = [screened_df['lat'].min(), screened_df['lat'].max()]
    for point in (comparison, barjamovic_2011, forlanini_2008, gcn_mean):
        if point is not None:
            xs.append(point[0]); ys.append(point[1])
    if minerals is not None and len(minerals) > 0:
        xs += minerals['lon'].tolist(); ys += minerals['lat'].tolist()
    pad = 0.15
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    # adjustable='datalim' (not the 'equal' default of adjustable='box'):
    # 'box' preserves the xlim/ylim exactly as set above but SHRINKS the
    # axes' on-screen rectangle to enforce equal degree scaling -- with a
    # tight local ellipse plus far-away comparison points, that shrinks
    # the plotted area down to a thin strip inside a mostly-blank figure,
    # reading as "overzoomed"/cramped even though nothing is technically
    # cut off. 'datalim' instead expands the shorter axis's data range to
    # fill the whole figure, so the full canvas is used and everything
    # (including far comparison points) has visible room around it.
    ax.set_aspect('equal', adjustable='datalim')

    # The exact ellipse (not the sparse candidate grid's own jagged edge)
    # -- built first so it can both clip the surface below and be drawn
    # as a crisp outline on top of it.
    ellipse_patch = None
    if ellipse_cov is not None and gcn_mean is not None:
        ellipse_patch = _ellipse_patch_from_cov(gcn_mean, ellipse_cov, ellipse_n_std)

    if len(grid_df) >= 3:
        tri_lon, tri_lat, tri_z = grid_df['lon'].to_numpy(), grid_df['lat'].to_numpy(), grid_df['rank_score'].to_numpy()
        if ellipse_patch is not None:
            # tricontourf triangulates whatever points it's given and fills
            # STRAIGHT-EDGED triangles between them -- with only ~n_per_axis^2
            # candidates, the outermost ones sit close to but strictly inside
            # the true ellipse, so the triangulated surface's own edge falls
            # visibly short of the smooth curve (worst at the widest points of
            # the ellipse, where the gap between a chord and the arc it
            # approximates is largest). Anchoring the triangulation with extra
            # points sampled ON the true ellipse -- each colored by its
            # nearest real candidate's rank_score, not a new/invented score --
            # extends the fill to the actual boundary without changing what
            # any real candidate is ranked. clip_path below remains the
            # authority on the surface never exceeding the ellipse; this is
            # purely about reaching it, not overshooting it.
            vals, vecs = np.linalg.eigh(np.asarray(ellipse_cov, dtype=float))
            theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            circle = np.stack([np.cos(theta), np.sin(theta)])
            offsets = (vecs * (ellipse_n_std * np.sqrt(np.maximum(vals, 1e-12)))) @ circle
            edge_lon = gcn_mean[0] + offsets[0]
            edge_lat = gcn_mean[1] + offsets[1]
            nearest_idx = np.array([
                np.argmin((tri_lon - lo) ** 2 + (tri_lat - la) ** 2)
                for lo, la in zip(edge_lon, edge_lat)
            ])
            tri_lon = np.concatenate([tri_lon, edge_lon])
            tri_lat = np.concatenate([tri_lat, edge_lat])
            tri_z = np.concatenate([tri_z, tri_z[nearest_idx]])
        cs = ax.tricontourf(tri_lon, tri_lat, tri_z,
                             levels=20, cmap='viridis_r', alpha=0.75, zorder=3)
        if ellipse_patch is not None:
            clip_patch = _ellipse_patch_from_cov(gcn_mean, ellipse_cov, ellipse_n_std)
            ax.add_patch(clip_patch)
            clip_patch.set_visible(False)  # only used as a clip boundary, not drawn itself
            try:
                cs.set_clip_path(clip_patch)
            except AttributeError:
                # Older Matplotlib: ContourSet exposes per-level collections
                # instead of behaving as a single clippable artist.
                for coll in cs.collections:
                    coll.set_clip_path(clip_patch)
        # Ordinal key, not a numeric scale: rank_score's actual values are
        # an arbitrary rank-sum with no standalone meaning (see
        # terrain.screen_ellipse()'s docstring) -- showing them as numbers
        # invited reading them as if they meant something on their own.
        # Two ticks only, relabeled Best/Worst, axis flipped so Best sits
        # at the top -- reads top-to-bottom like a ranked list instead of
        # a quantity scale.
        cbar = fig.colorbar(cs, ax=ax, shrink=0.7)
        cbar.ax.invert_yaxis()
        cbar.set_ticks([grid_df['rank_score'].min(), grid_df['rank_score'].max()])
        cbar.ax.set_yticklabels(['Best', 'Worst'])
    elif len(grid_df) > 0:
        # Too few points to triangulate a contour surface -- fall back to
        # plain scatter rather than silently drop the data.
        ax.scatter(grid_df['lon'], grid_df['lat'], c=grid_df['rank_score'], cmap='viridis_r',
                   s=40, alpha=0.85, zorder=3, edgecolors='none')

    if ellipse_patch is not None:
        ellipse_patch.set(facecolor='none', edgecolor='red', linewidth=1.5, linestyle='-',
                           zorder=4, label=f'{ellipse_n_std:g}-Sigma boundary')
        ax.add_patch(ellipse_patch)

    if len(site_df) > 0:
        ax.scatter(site_df['lon'], site_df['lat'], c='orange', marker='^', s=110,
                   edgecolors='k', zorder=6, label='Known site')
        for _, row in site_df.iterrows():
            ax.annotate(row['known_site'], (row['lon'], row['lat']), fontsize=7,
                        color='darkorange', xytext=(5, -8), textcoords='offset points', zorder=6,
                        path_effects=_TEXT_HALO)

    if minerals is not None and len(minerals) > 0:
        ax.scatter(minerals['lon'], minerals['lat'], c='mediumpurple', marker='D', s=70,
                   edgecolors='k', zorder=6, label='Mineral Deposit')
        for _, ore_row in minerals.iterrows():
            # No _TEXT_HALO here (unlike known-site labels below): at this
            # smaller fontsize the black stroke overwhelmed the thin
            # indigo text rather than helping it -- illegible instead of
            # readable.
            ax.annotate(f"{ore_row['locality']} ({ore_row['metal']})", (ore_row['lon'], ore_row['lat']),
                        fontsize=6, color='indigo', xytext=(4, -8), textcoords='offset points', zorder=6)

    if comparison is not None:
        ax.scatter(*comparison, c='green', marker='s', s=80, edgecolors='k',
                   label=comparison_label, zorder=4)
    if barjamovic_2011 is not None:
        ax.scatter(*barjamovic_2011, marker='$B$', s=170, color='darkgreen', linewidth=0.4,
                   label='Barjamovic 2011 (historian)', zorder=6)
    if forlanini_2008 is not None:
        ax.scatter(*forlanini_2008, marker='$F$', s=170, color='darkred', linewidth=0.4,
                   label='Forlanini 2008 (historian)', zorder=6)
    if gcn_mean is not None:
        mean_label = f"{_model_family(model_label)[0]} Mean Prediction"
        ax.scatter(*gcn_mean, c='black', marker='x', s=100, label=mean_label, zorder=6)

    ax.set_title(
        f"Geoscreened Candidate Sites ({model_label}): {name}"
        + (f"\n({len(site_df)} known site(s) in range)" if len(site_df) else "")
    )
    # markerscale shrinks the legend's icons independent of their actual
    # on-map size ('B'/'F' at s=170 vs. GCN Mean at s=100 etc. otherwise
    # render at wildly different icon sizes in the legend box and visibly
    # overlap each other); labelspacing/handletextpad/borderpad give each
    # row breathing room so entries don't crowd together.
    ax.legend(loc='lower left', fontsize=8, markerscale=0.6, labelspacing=0.7,
              handletextpad=0.6, borderpad=0.6)
    # No plt.tight_layout() here (unlike the other plot functions in this
    # module): this is the only figure combining a basemap image AND a
    # fig.colorbar() -- that combination's subplot-margin reconciliation
    # crashes natively (Fatal Python error: Aborted, inside matplotlib's
    # own get_tick_space()/get_affine() during tight_layout()'s bbox
    # computation) in this environment. tight_layout() is purely cosmetic
    # margin spacing, so it's simply skipped for this plot rather than
    # worked around -- the figure is still complete and correct, just not
    # auto-trimmed.

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # bbox_inches='tight' is a post-render crop of the finished figure
        # (computed from the rendered artists' actual extents), a
        # different code path from plt.tight_layout()'s pre-render
        # subplot-margin adjustment -- the crash documented above is
        # specific to tight_layout() with this figure's basemap+colorbar
        # combination, not to this. If it does turn out to crash here too
        # when rerun, drop bbox_inches/pad_inches from this call only.
        fig.savefig(os.path.join(save_dir, _build_filename(model_label, category, name)),
                    dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)
    return fig


def plot_aggregate_map(names, results, model_label, category, actual_coords_by_name=None,
                        comparison_coords_by_name=None, comparison_label='Barjamovic et al. (gravity model)',
                        reference_coords_by_name=None, reference_label='Known city',
                        overall_error_km=None, save_dir=None, shapefile_path=None,
                        xlim=(31, 39.3), ylim=(35.8, 42.2)):
    """category: see plot_individual_uncertainty()'s docstring -- used
    only to build the saved filename.
    Deliberately does NOT take barjamovic_2011/forlanini_2008 params --
    10 cities x 3 extra markers each would make this map unreadable.
    Those two historians' proposals are shown only on the per-city plots
    (plot_individual_uncertainty above).

    reference_coords_by_name: optional {name: (lon, lat)}, plotted as
    plain background markers for geographic orientation (e.g. the 15
    known ancient cities, shown for context on a lost-city map) -- NO
    line is drawn to anything. This is different in kind from
    actual_coords_by_name/comparison_coords_by_name, which both draw a
    dashed line to the GCN mean specifically to show error/disagreement
    against a claim about THAT city; a reference city isn't a claim
    about any of the plotted predictions, just orientation."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    load_and_plot_turkey_map(ax, shapefile_path)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')

    if reference_coords_by_name is not None:
        for idx, (ref_name, (ref_lon, ref_lat)) in enumerate(reference_coords_by_name.items()):
            ax.scatter(ref_lon, ref_lat, c='blue', marker='o', s=80, edgecolors='k',
                       zorder=4, label=reference_label if idx == 0 else "")
            ax.text(ref_lon, ref_lat + 0.03, ref_name, fontsize=7, color='blue',
                    ha='center', zorder=4)

    for idx, name in enumerate(names):
        pts = np.asarray(results[name])
        mean_c = pts.mean(axis=0)

        if actual_coords_by_name is not None:
            actual = actual_coords_by_name[name]
            ax.scatter(actual[0], actual[1], c='blue', marker='o', s=80,
                       label='Actual' if idx == 0 else "", edgecolors='k', zorder=5)
            ax.plot([actual[0], mean_c[0]], [actual[1], mean_c[1]], 'r--', alpha=0.4)
            ax.text(actual[0], actual[1] + 0.03, name, fontsize=8, color='blue', ha='center')
        else:
            ax.text(mean_c[0], mean_c[1] + 0.05, name, fontsize=8, color='red', fontweight='bold')

        mean_label = f"{_model_family(model_label)[0]} Mean"
        ax.scatter(mean_c[0], mean_c[1], c='red', marker='X', s=80,
                   label=mean_label if idx == 0 else "", edgecolors='k', zorder=5)

        if comparison_coords_by_name is not None:
            comp = comparison_coords_by_name[name]
            ax.scatter(comp[0], comp[1], c='green', marker='s', s=80,
                       label=comparison_label if idx == 0 else "", edgecolors='k', zorder=4)
            ax.plot([mean_c[0], comp[0]], [mean_c[1], comp[1]], 'k--', alpha=0.4)

    title = f"{model_label}: Bootstrapped {_model_family(model_label)[0]} Means vs. Actual"
    if overall_error_km is not None:
        title += f"\nOverall Mean Error: {overall_error_km:.2f} km"
    ax.set_title(title, fontsize=16)
    ax.legend(loc='lower left', fontsize=8)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # ax.set_aspect('equal') above (default adjustable='box') shrinks
        # the axes rectangle to enforce equal lon/lat scaling within the
        # fixed FIGSIZE canvas -- since xlim/ylim's ~8.3x6.4 degree extent
        # doesn't match FIGSIZE's ~12x10 aspect ratio, that shrink used to
        # leave blank bands of literal white canvas around the map (most
        # visible once placed in LaTeX, where the figure gets scaled to a
        # fixed width and the blank margin eats into the apparent size of
        # the actual content). bbox_inches='tight' crops the saved file
        # down to the rendered content's true extent, removing that
        # margin regardless of the aspect-ratio mismatch that caused it.
        fig.savefig(os.path.join(save_dir, _build_filename(model_label, category)),
                    dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)
    return fig
