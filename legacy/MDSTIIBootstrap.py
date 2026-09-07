# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:30:00 2026
@author: user
TII Normalization: Comprehensive Bootstrap Analysis, Tables, and Multi-Plotting
"""

import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import geopandas
import contextily as cx
from tqdm import tqdm

# --- Configuration ---
SHAPEFILE_PATH = r'C:\Users\user\OneDrive\Documents\Work\25 Fall\IE490\Codes\gadm41_TUR_shp\gadm41_TUR_1.shp' 
N_BOOTSTRAP = 40

# --- Helper Functions ---

def calculate_euclidean_distance_km(actual_long, actual_lat, pred_long, pred_lat):
    cos_factor = np.cos(np.radians(37.9))
    y_diff = actual_lat - pred_lat
    x_diff = actual_long - pred_long
    return (10000 / 90) * np.sqrt(y_diff**2 + (cos_factor * x_diff)**2)

def draw_confidence_ellipse(pts, ax, n_std=1.0, edgecolor='red', label=None, **kwargs):
    mean = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell = Ellipse(xy=mean, width=width, height=height, angle=theta,
                  edgecolor=edgecolor, facecolor='none', label=label, **kwargs)
    ax.add_patch(ell)
    return ell

def load_and_plot_turkey_map(ax):
    try:
        turkey_map = geopandas.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
        turkey_map.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, alpha=0.4, zorder=2)
    except Exception: pass
    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenTopoMap, zorder=0, alpha=0.5)
    except Exception: pass

# --- Step 1: Data Setup ---
traffic_data = {
    'City1': ['Durhumit', 'Durhumit', 'Durhumit', 'Durhumit', 'Durhumit', 'Durhumit', 'Durhumit', 'Durhumit', 'Hahhum', 'Hahhum', 'Hahhum', 'Hahhum', 'Hahhum', 'Hahhum', 'Hahhum', 'Hahhum', 'Hanaknak', 'Hanaknak', 'Hattus', 'Hattus', 'Hurama', 'Hurama', 'Hurama', 'Hurama', 'Hurama', 'Hurama', 'Hurama', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kanes', 'Kuburnat', 'Kuburnat', 'Malitta', 'Mamma', 'Mamma', 'Mamma', 'Ninassa', 'Ninassa', 'Ninassa', 'Ninassa', 'Purushaddum', 'Purushaddum', 'Purushaddum', 'Purushaddum', 'Purushaddum', 'Purushaddum', 'Salatuwar', 'Salatuwar', 'Salatuwar', 'Salatuwar', 'Salatuwar', 'Salatuwar', 'Samuha', 'Samuha', 'Samuha', 'Sinahuttum', 'Sinahuttum', 'Sinahuttum', 'Sinahuttum', 'Suppiluliya', 'Tapaggas', 'Timelkiya', 'Timelkiya', 'Timelkiya', 'Timelkiya', 'Timelkiya', 'Timelkiya', 'Timelkiya', 'Timelkiya', 'Timelkiya', 'Tuhpiya', 'Tuhpiya', 'Tuhpiya', 'Tuhpiya', 'Tuhpiya', 'Ulama', 'Ulama', 'Ulama', 'Ulama', 'Unipsum', 'Unipsum', 'Wahsusana', 'Wahsusana', 'Wahsusana', 'Wahsusana', 'Wahsusana', 'Wahsusana', 'Wahsusana', 'Wahsusana', 'Wahsusana', 'Wahsusana', 'Wahsusana', 'Washaniya', 'Washaniya', 'Washaniya', 'Washaniya', 'Zalpa', 'Zalpa', 'Zalpa', 'Zimishuna', 'Zimishuna'],
    'City2': ['Kanes', 'Purushaddum', 'Salatuwar', 'Sinahuttum', 'Timelkiya', 'Ulama', 'Wahsusana', 'Zimishuna', 'Hanaknak', 'Hattus', 'Kanes', 'Malitta', 'Purushaddum', 'Timelkiya', 'Wahsusana', 'Zimishuna', 'Tapaggas', 'Zalpa', 'Sinahuttum', 'Suppiluliya', 'Hanaknak', 'Hattus', 'Kanes', 'Kuburnat', 'Salatuwar', 'Timelkiya', 'Wahsusana', 'Hahhum', 'Hanaknak', 'Hurama', 'Kuburnat', 'Malitta', 'Ninassa', 'Purushaddum', 'Salatuwar', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana', 'Washaniya', 'Zalpa', 'Hanaknak', 'Malitta', 'Wahsusana', 'Karahna', 'Unipsum', 'Zalpa', 'Purushaddum', 'Ulama', 'Wahsusana', 'Washaniya', 'Durhumit', 'Hanaknak', 'Hattus', 'Salatuwar', 'Timelkiya', 'Ulama', 'Hanaknak', 'Ninassa', 'Purushaddum', 'Timelkiya', 'Wahsusana', 'Zimishuna', 'Hattus', 'Karahna', 'Kuburnat', 'Durhumit', 'Hattus', 'Timelkiya', 'Zimishuna', 'Karahna', 'Hanaknak', 'Hanaknak', 'Hattus', 'Hurama', 'Kuburnat', 'Purushaddum', 'Salatuwar', 'Tuhpiya', 'Wahsusana', 'Washaniya', 'Durhumit', 'Kanes', 'Salatuwar', 'Timelkiya', 'Ulama', 'Hattus', 'Ninassa', 'Purushaddum', 'Salatuwar', 'Hattus', 'Kuburnat', 'Durhumit', 'Hattus', 'Kanes', 'Ninassa', 'Purushaddum', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Washaniya', 'Zimishuna', 'Malitta', 'Purushaddum', 'Timelkiya', 'Ulama', 'Hattus', 'Kanes', 'Timelkiya', 'Purushaddum', 'Sinahuttum'],
    'Traffic': [3, 13, 5, 1, 1, 2, 10, 1, 1, 8, 18, 1, 2, 19, 1, 4, 1, 4, 1, 2, 2, 15, 1, 4, 1, 3, 2, 1, 7, 2, 1, 1, 3, 3, 2, 4, 2, 2, 12, 5, 1, 6, 1, 2, 5, 1, 1, 2, 2, 3, 2, 6, 1, 1, 5, 1, 2, 1, 1, 8, 3, 7, 1, 1, 1, 1, 2, 3, 1, 1, 2, 1, 3, 21, 19, 4, 3, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, 5, 1, 3, 5, 5, 1, 3, 1, 1, 19, 1, 7, 1, 13, 1, 1, 4, 1, 1, 1, 6, 2, 1, 1]
}
traffic_df = pd.DataFrame(traffic_data)

cities_df = pd.DataFrame({
    'name': ['Hattus', 'Kanes', 'Karahna', 'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta', 'Salatuwar', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana', 'Zimishuna', 'Mamma'],
    'lat_y': [40.021, 38.85, 40.0, 40.148, 40.0, 38.261, 39.363, 39.655, 39.619, 38.027, 38.411, 38.021, 39.584, 40.461, 37.583],
    'long_x': [34.61, 35.633, 36.1, 35.762, 35.817, 37.114, 33.787, 31.994, 36.528, 38.234, 33.834, 36.503, 33.418, 35.65, 36.933]
}).set_index('name')

bajramovic_df = pd.DataFrame({
    'name': ['Hattus', 'Kanes', 'Karahna', 'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta', 'Mamma', 'Salatuwar', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana', 'Zimishuna'],
    'est_lat': [39.997, 39.313, 40.046, 40.0, 40.15, 39.139, 38.888, 38.02, 39.561, 38.3, 38.261, 39.835, 37.583, 39.003, 39.234],
    'est_long': [36.131, 33.918, 34.547, 35.817, 35.761, 38.226, 35.282, 36.503, 33.356, 37.118, 37.114, 33.234, 36.933, 31.926, 34.213]
}).set_index('name')

all_cities = cities_df.index.tolist()

# --- Step 2: Bootstrap Loop ---
boot_results = {city: [] for city in all_cities}

print(f"\nStarting {N_BOOTSTRAP} Bootstrap Runs with TII Normalization...")
for b in tqdm(range(N_BOOTSTRAP)):
    b_traffic = traffic_df.copy()
    b_traffic['Traffic'] = np.random.poisson(b_traffic['Traffic']).astype(float)
    b_traffic.loc[b_traffic['Traffic'] == 0, 'Traffic'] = 0.5

    d_mat = pd.DataFrame(0.0, index=all_cities, columns=all_cities)
    for _, row in b_traffic.iterrows():
        if row['City1'] in all_cities and row['City2'] in all_cities:
            d_mat.loc[row['City1'], row['City2']] += row['Traffic']

    total_exports = d_mat.sum(axis=1)
    total_imports = d_mat.sum(axis=0)
    world_trade = d_mat.sum().sum()

    T_ij = (d_mat * world_trade) / (np.outer(total_exports, total_imports) + 1e-9)
    T_ji = (d_mat.T * world_trade) / (np.outer(total_imports, total_exports) + 1e-9)
    b_sim = 0.5 * (T_ij + T_ji)

    with np.errstate(divide='ignore'):
        b_dissim = np.sqrt(1.0 / (b_sim + 1e-9))
    b_dissim = b_dissim.replace([np.inf, -np.inf], 100.0)
    np.fill_diagonal(b_dissim.values, 0)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=b, n_init=1)
    b_mds_coords = mds.fit_transform(b_dissim)
    b_mds_df = pd.DataFrame(b_mds_coords, index=all_cities, columns=['x', 'y'])

    for city_to_predict in all_cities:
        train_cities = [c for c in all_cities if c != city_to_predict]
        known_real = cities_df.loc[train_cities, ['long_x', 'lat_y']].to_numpy()
        known_mds = b_mds_df.loc[train_cities].to_numpy()
        
        w_vec = b_sim.loc[city_to_predict, train_cities].to_numpy()
        w_vec = (w_vec - w_vec.min()) / (w_vec.max() - w_vec.min() + 1e-9) + 0.1
        W = np.diag(w_vec)

        c1 = np.average(known_real, axis=0, weights=w_vec)
        c2 = np.average(known_mds, axis=0, weights=w_vec)
        A = (known_real - c1).T @ W @ (known_mds - c2)
        U, _, Vt = np.linalg.svd(A)
        R = Vt.T @ U.T
        scale = np.linalg.norm(known_real - c1) / np.linalg.norm(known_mds - c2)
        
        est = (b_mds_df.loc[city_to_predict].to_numpy() - c2) @ R.T * scale + c1
        boot_results[city_to_predict].append(est)

# --- Step 3: Result Aggregation ---
final_data = []
for city in all_cities:
    coords = np.array(boot_results[city])
    m_long, m_lat = coords.mean(axis=0)
    s_long, s_lat = coords.std(axis=0)
    error = calculate_euclidean_distance_km(cities_df.loc[city, 'long_x'], cities_df.loc[city, 'lat_y'], m_long, m_lat)
    final_data.append({
        'city': city, 'actual_lat': cities_df.loc[city, 'lat_y'], 'mean_lat': m_lat, 'std_lat': s_lat,
        'actual_long': cities_df.loc[city, 'long_x'], 'mean_long': m_long, 'std_long': s_long, 'error_km': error
    })

results_df = pd.DataFrame(final_data)
city_print_order = ['Hattus', 'Kanes', 'Karahna', 'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta', 'Mamma', 'Salatuwar', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana', 'Zimishuna']
results_df = results_df.set_index('city').reindex(city_print_order).reset_index()

# --- Display Final Table ---
print("\n" + "="*125)
print(f"FINAL BOOTSTRAP RESULTS ({N_BOOTSTRAP} runs - Poisson)")
print("="*125)
table_cols = ['city', 'actual_lat', 'actual_long', 'mean_long', 'mean_lat','std_long', 'std_lat', 'error_km']
print(results_df[table_cols].to_string(index=False, float_format="%.3f"))
print(f"\nOverall Average Error: {results_df['error_km'].mean():.2f} km")

fig, ax = plt.subplots(figsize=(14, 10))
load_and_plot_turkey_map(ax)
ax.set_xlim(31, 39.3); ax.set_ylim(35.8, 42.2); ax.set_aspect('equal')

# Main Markers
ax.scatter(results_df['actual_long'], results_df['actual_lat'], c='blue', marker='o', s=100, label='Actual', edgecolors='k', zorder=5)
ax.scatter(results_df['mean_long'], results_df['mean_lat'], c='red', marker='X', s=100, label='TII Est.', edgecolors='k', zorder=5)
ax.scatter(bajramovic_df['est_long'], bajramovic_df['est_lat'], c='green', marker='s', s=80, label='Bajramovic Est.', edgecolors='k', zorder=4)

# Error Lines
for _, row in results_df.iterrows():
    ax.plot([row['actual_long'], row['mean_long']], [row['actual_lat'], row['mean_lat']], 'r--', alpha=0.5)
    ax.text(row['actual_long'], row['actual_lat'] + 0.05, row['city'], fontsize=8, color='blue')

for name, row in bajramovic_df.iterrows():
    if name in cities_df.index:
        act = cities_df.loc[name]
        ax.plot([act['long_x'], row['est_long']], [act['lat_y'], row['est_lat']], 'g:', alpha=0.6)

ax.set_title(f'TII Model Comparison: Actual vs. MDS vs. Bajramovic\nTII Mean Error: {results_df["error_km"].mean():.2f} km', fontsize=16)
ax.legend(loc='lower left')
plt.tight_layout()
plt.show()

# --- Step 5: 15 Individual Plots & Detailed Tables ---
print("\nGenerating Individual Results and Confidence Plots...")
for name in city_print_order:
    # 1. Individual Table
    city_metrics = results_df[results_df['city'] == name]
    
    # 2. Confidence Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    load_and_plot_turkey_map(ax)
    pts = np.array(boot_results[name])
    
    # Bootstrap cloud and Ellipses
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=15, color='gray', label='Bootstrap Pts', zorder=3)
    draw_confidence_ellipse(pts, ax, n_std=1.0, edgecolor='blue', linewidth=2, label='1-Sigma', zorder=4)
    draw_confidence_ellipse(pts, ax, n_std=2.0, edgecolor='red', linestyle='--', linewidth=1.5, label='2-Sigma', zorder=4)
    
    # Reference Markers
    ax.scatter(city_metrics['mean_long'], city_metrics['mean_lat'], c='red', marker='X', s=120, label='TII Mean', edgecolors='k', zorder=6)
    ax.scatter(city_metrics['actual_long'], city_metrics['actual_lat'], c='blue', marker='o', s=100, label='Actual', edgecolors='k', zorder=6)
    
    if name in bajramovic_df.index:
        baj = bajramovic_df.loc[name]
        ax.scatter(baj['est_long'], baj['est_lat'], c='green', marker='s', s=80, label='Bajramovic', edgecolors='k', zorder=5)

    # Zoom Logic
    pad = 0.8
    all_x = [pts[:,0].min(), pts[:,0].max(), city_metrics['actual_long'].values[0]]
    all_y = [pts[:,1].min(), pts[:,1].max(), city_metrics['actual_lat'].values[0]]
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    
    ax.set_title(f"Position Confidence: {name} (TII Error: {city_metrics['error_km'].values[0]:.1f}km)")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()