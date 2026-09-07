# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 18:12:22 2026
MDS Prediction for 10 Unknown Cities (Updated Legend and Scaling)
@author: user
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
MASTER_SEED = 42
N_BOOTSTRAP = 200
SHAPEFILE_PATH = r'C:\Users\user\OneDrive\Documents\Work\25 Fall\IE490\Codes\gadm41_TUR_shp\gadm41_TUR_1.shp'

np.random.seed(MASTER_SEED)
BOOTSTRAP_SEEDS = np.random.randint(0, 10000, size=N_BOOTSTRAP)

# --- Helper Functions ---
def calculate_distance_km(p1_long, p1_lat, p2_long, p2_lat):
    cos_factor = np.cos(np.radians(37.9))
    y_diff = p1_lat - p2_lat
    x_diff = p1_long - p2_long
    return (10000 / 90) * np.sqrt(y_diff**2 + (cos_factor * x_diff)**2)

def draw_confidence_ellipse(pts, ax, n_std=1.0, edgecolor='red', label=None, **kwargs):
    mean = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))
    ell = Ellipse(xy=mean, width=width, height=height, angle=theta,
                  edgecolor=edgecolor, facecolor='none', label=label, **kwargs)
    ax.add_patch(ell)
    return ell

def load_and_plot_turkey_map(ax):
    try:
        turkey_map = geopandas.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
        turkey_map.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, alpha=0.4, zorder=2)
    except: pass
    try: cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenTopoMap, zorder=0, alpha=0.5)
    except: pass

# --- Data Setup ---
all_city_names = [
    'Durhumit', 'Hahhum', 'Hanaknak', 'Hattus', 'Hurama', 'Kanes', 'Karahna',
    'Kuburnat', 'Malitta', 'Mamma', 'Ninassa', 'Purushaddum', 'Salatuwar',
    'Samuha', 'Sinahuttum', 'Suppiluliya', 'Tapaggas', 'Timelkiya', 'Tuhpiya',
    'Ulama', 'Unipsum', 'Wahsusana', 'Washaniya', 'Zalpa', 'Zimishuna'
]

matrix_data = np.array([
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 13, 5, 0, 1, 0, 0, 1, 2, 0, 0, 10, 0, 0, 1], [1, 0, 0, 0, 0, 8, 18, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 19, 0, 0, 1, 4, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], 
    [0, 2, 0, 0, 0, 15, 1, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0], [0, 7, 0, 0, 0, 0, 2, 1, 2, 0, 3, 3, 2, 0, 0, 0, 0, 4, 2, 2, 0, 12, 5, 1, 0], 
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0], [6, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 3, 0, 0, 0, 7, 0, 1, 0], [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 1, 0, 21, 19, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0], 
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [5, 0, 0, 1, 0, 3, 0, 0, 0, 0, 1, 22, 19, 0, 0, 0, 1, 1, 7, 0, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 2, 1, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
])

known_coords = {
    'Hattus': [40.021, 34.61], 'Kanes': [38.85, 35.633], 'Karahna': [40.0, 36.1], 
    'Tapaggas': [40.148, 35.762], 'Hanaknak': [40.0, 35.817], 'Hurama': [38.261, 37.114], 
    'Malitta': [39.363, 33.787], 'Salatuwar': [39.655, 31.994], 'Samuha': [39.619, 36.528], 
    'Timelkiya': [38.027, 38.234], 'Ulama': [38.411, 33.834], 'Unipsum': [38.021, 36.503], 
    'Wahsusana': [39.584, 33.418], 'Zimishuna': [40.461, 35.65], 'Mamma': [37.583, 36.933]
}

bajramovic_unknown = {
    'Durhumit': [40.47, 35.65], 'Hahhum': [38.43, 38.04], 'Kuburnat': [40.71, 36.52], 
    'Ninassa': [38.98, 34.61], 'Purushaddum': [39.71, 32.87], 'Sinahuttum': [39.96, 34.87], 
    'Suppiluliya': [40.02, 34.62], 'Tuhpiya': [39.61, 35.2], 'Washaniya': [39.16, 34.31], 'Zalpa': [38.81, 37.86]
}

lost_cities = list(bajramovic_unknown.keys())
known_cities = list(known_coords.keys())
bootstrap_results = {city: [] for city in lost_cities}

# --- Bootstrap Training Loop ---
print(f"Sampling MDS Predictions for {len(lost_cities)} lost cities (N={N_BOOTSTRAP})...")

for seed in tqdm(BOOTSTRAP_SEEDS):
    b_matrix = np.random.poisson(matrix_data).astype(float)
    b_matrix[b_matrix == 0] = 0.5
    
    col_sums = b_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1e-9
    S_matrix = b_matrix / col_sums
    I_matrix = 0.5 * (S_matrix + S_matrix.T)
    
    b_dissim = np.sqrt(1.0 / (I_matrix + 1e-9))
    np.fill_diagonal(b_dissim, 0)
    
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=int(seed), n_init=1)
    coords_2d = mds.fit_transform(b_dissim)
    df_mds = pd.DataFrame(coords_2d, index=all_city_names, columns=['x', 'y'])
    
    Y_known = np.array([known_coords[c] for c in known_cities])
    X_known = df_mds.loc[known_cities].to_numpy()
    
    c_y = Y_known.mean(axis=0)
    c_x = X_known.mean(axis=0)
    
    A = (Y_known - c_y).T @ (X_known - c_x)
    U, _, Vt = np.linalg.svd(A)
    R = Vt.T @ U.T
    scale = np.linalg.norm(Y_known - c_y) / np.linalg.norm(X_known - c_x)
    
    X_lost = df_mds.loc[lost_cities].to_numpy()
    Y_pred = (X_lost - c_x) @ R.T * scale + c_y
    
    for i, city in enumerate(lost_cities):
        bootstrap_results[city].append([Y_pred[i, 1], Y_pred[i, 0]])

# --- Output Generation ---

# A. Table
table_rows = []
for name in lost_cities:
    pts = np.array(bootstrap_results[name])
    mean_c = np.mean(pts, axis=0)
    std_c = np.std(pts, axis=0)
    baj = bajramovic_unknown[name]
    dist = calculate_distance_km(mean_c[0], mean_c[1], baj[1], baj[0])
    table_rows.append({
        'City': name, 'Mean_Long': mean_c[0], 'Mean_Lat': mean_c[1], 
        'Std_Long': std_c[0], 'Std_Lat': std_c[1], 'Dist_to_Baj': dist
    })

results_df = pd.DataFrame(table_rows)
print("\n--- PREDICTION RESULTS (MDS DYADIC) ---")
print(results_df.round(3).to_string(index=False))

# B. Individual Confidence Plots (FIXED LEGEND AND LIMITS)
for name in lost_cities:
    fig, ax = plt.subplots(figsize=(8, 6))
    load_and_plot_turkey_map(ax)
    pts = np.array(bootstrap_results[name])
    baj_lat, baj_long = bajramovic_unknown[name]
    
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=15, color='gray')
    draw_confidence_ellipse(pts, ax, n_std=1.0, edgecolor='blue', label='1-Sigma')
    draw_confidence_ellipse(pts, ax, n_std=2.0, edgecolor='red', label='2-Sigma')
    
    mean_long, mean_lat = np.mean(pts, axis=0)
    ax.scatter(mean_long, mean_lat, c='red', marker='X', s=100, label='MDS Mean')
    ax.scatter(baj_long, baj_lat, c='green', marker='s', s=80, label='Barjamovic')
    
    # 1. Force equal aspect ratio so ellipses aren't squashed
    ax.set_aspect('equal')
    
    # 2. Update limits AFTER ellipses are drawn so they fit
    pad = 1.0  # Increased pad to prevent clipping
    ax.set_xlim(min(pts[:,0].min(), baj_long) - pad, max(pts[:,0].max(), baj_long) + pad)
    ax.set_ylim(min(pts[:,1].min(), baj_lat) - pad, max(pts[:,1].max(), baj_lat) + pad)
    
    ax.set_title(f"Prediction Confidence: {name}")
    
    # 3. Move Legend to upper right to avoid overlap
    ax.legend(loc='upper right', frameon=True, fontsize='small')
    plt.show()

# C. Overall Map
fig, ax = plt.subplots(figsize=(12, 10))
load_and_plot_turkey_map(ax)
ax.set_xlim(31, 39.3)
ax.set_ylim(35.8, 42.2)
ax.set_aspect('equal')

for name in lost_cities:
    pts = np.array(bootstrap_results[name])
    mlong, mlat = np.mean(pts, axis=0)
    blat, blong = bajramovic_unknown[name]
    ax.scatter(mlong, mlat, c='red', marker='X', s=100, zorder=5)
    ax.scatter(blong, blat, c='green', marker='s', s=80, zorder=5)
    ax.plot([mlong, blong], [mlat, blat], 'k--', alpha=0.5)
    ax.text(mlong, mlat + 0.05, name, fontsize=8, color='red', fontweight='bold')

ax.set_title(f"Final MDS Predictions vs Barjamovic\nAvg Distance: {results_df['Dist_to_Baj'].mean():.2f} km")
plt.show()