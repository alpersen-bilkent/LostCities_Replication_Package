# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:15:02 2026
Modern Turkish Cities - Unified Methodology with Poisson Scaling
@author: user
"""
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import geopandas as gpd
import contextily as cx
from tqdm import tqdm
import os

# --- 1. CONFIGURATION ---
FILE_PATH = 'C:/Users/user/OneDrive/Documents/Work/25 Fall/IE490/İller Arası Ticaret.xlsx'
SHAPEFILE_PATH = r'C:\Users\user\OneDrive\Documents\Work\25 Fall\IE490\Codes\gadm41_TUR_shp\gadm41_TUR_1.shp'
N_BOOTSTRAP = 40

# --- 2. COORDINATES & HELPER FUNCTIONS ---
coords_dict = {
    'ADIYAMAN': (37.7648, 38.2786), 'AKSARAY': (38.3687, 34.0370), 
    'AMASYA': (40.6500, 35.8300), 'ÇANKIRI': (40.6013, 33.6134),
    'ÇORUM': (40.5506, 34.9556), 'KAHRAMANMARAŞ': (37.5710, 36.9371),
    'KAYSERİ': (38.7312, 35.4787), 'KIRIKKALE': (39.8468, 33.5153), 
    'KIRŞEHİR': (39.1425, 34.1709), 'MALATYA': (38.3552, 38.3095),
    'NEVŞEHİR': (38.6244, 34.7144), 'NİĞDE': (37.9667, 34.6833),
    'SİVAS': (39.7477, 37.0179), 'TOKAT': (40.3167, 36.5500),
    'YOZGAT': (39.8181, 34.8147),
}
cities_df = pd.DataFrame.from_dict(coords_dict, orient='index', columns=['lat_y', 'long_x'])

def calculate_euclidean_distance_km(actual_long, actual_lat, pred_long, pred_lat):
    # Standardized to 37.9 to match Ancient City script
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
    if os.path.exists(SHAPEFILE_PATH):
        try:
            turkey_map = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
            turkey_map.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, alpha=0.4, zorder=2)
        except Exception: pass
    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenTopoMap, zorder=0, alpha=0.5)
    except Exception: pass

# --- 3. DATA LOADING & PREPROCESSING ---
print(f"Attempting to load: {FILE_PATH}")
df_raw = pd.read_excel(FILE_PATH, header=0) if FILE_PATH.endswith('.xlsx') else pd.read_csv(FILE_PATH, header=0)
df_raw.rename(columns={df_raw.columns[0]: 'City'}, inplace=True)
df_raw.set_index('City', inplace=True)
df_raw = df_raw.replace({'*': 0, '-': 0}).infer_objects(copy=False) 
df_raw = df_raw.apply(pd.to_numeric, errors='coerce').fillna(0)

final_cities = df_raw.index.intersection(cities_df.index)
df_main = df_raw.loc[final_cities, final_cities]
cities_df = cities_df.loc[final_cities]
all_cities_list = final_cities.tolist()
print(f"✅ Data processed for {len(all_cities_list)} cities.")

# --- 4. BOOTSTRAP LOOP (Unified with Scaling) ---
boot_results = {city: [] for city in all_cities_list}

# Prevent lam too large: target max lambda of 1,000,000
max_val = df_main.values.max()
scaling_factor = 1e3 / max_val if max_val > 1e3 else 1.0

print(f"\nStarting {N_BOOTSTRAP} Bootstrap Runs with Poisson Perturbations...")
for b in tqdm(range(N_BOOTSTRAP)):
    # 1. Poisson Generation on Scaled Data
    b_vol = np.random.poisson(df_main.values * scaling_factor).astype(float) / scaling_factor
    b_vol[b_vol == 0] = 0.5 

    # 2. Dyadic Similarity (Consistent with Ancient Code)
    row_sums = b_vol.sum(axis=1)
    col_sums = b_vol.sum(axis=0)
    total_act = row_sums + col_sums
    
    with np.errstate(divide='ignore', invalid='ignore'):
        term_i = b_vol / total_act[:, np.newaxis]
        term_j = b_vol / total_act[np.newaxis, :]
        b_sim_arr = 0.5 * (term_i + term_j)
        b_sim_arr = b_sim_arr + b_sim_arr.T
        b_sim_arr = np.nan_to_num(b_sim_arr, nan=1e-9)
        
        # 3. Dissimilarity Inversion
        b_dissim_arr = np.sqrt(1.0 / (b_sim_arr + 1e-12))
        np.fill_diagonal(b_dissim_arr, 0)

    b_dissim = pd.DataFrame(b_dissim_arr, index=all_cities_list, columns=all_cities_list)
    b_sim = pd.DataFrame(b_sim_arr, index=all_cities_list, columns=all_cities_list)

    # 4. MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=b, n_init=1)
    b_mds_coords = mds.fit_transform(b_dissim)
    b_mds_df = pd.DataFrame(b_mds_coords, index=all_cities_list, columns=['x', 'y'])

    # 5. Weighted Procrustes Alignment (Unified Logic)
    for city_to_predict in all_cities_list:
        train_cities = [c for c in all_cities_list if c != city_to_predict]
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
        
        mds_point = b_mds_df.loc[city_to_predict].to_numpy()
        est = (mds_point - c2) @ R.T * scale + c1
        boot_results[city_to_predict].append(est)

# --- 5. AGGREGATION & TABLE ---
final_data = []
for city in all_cities_list:
    pts = np.array(boot_results[city])
    m_long, m_lat = pts.mean(axis=0)
    error = calculate_euclidean_distance_km(cities_df.loc[city, 'long_x'], cities_df.loc[city, 'lat_y'], m_long, m_lat)
    final_data.append({
        'city': city, 'actual_lat': cities_df.loc[city, 'lat_y'], 'actual_long': cities_df.loc[city, 'long_x'],
        'mean_lat': m_lat, 'mean_long': m_long, 'std_long': pts[:,0].std(), 'std_lat': pts[:,1].std(), 'error_km': error
    })

results_df = pd.DataFrame(final_data)
# Applying same sorting/printing logic as Ancient code
modern_print_order = ['SİVAS', 'KAYSERİ', 'MALATYA', 'KAHRAMANMARAŞ', 'ADIYAMAN', 'TOKAT', 'AMASYA', 'ÇORUM', 'YOZGAT', 'KIRŞEHİR', 'NEVŞEHİR', 'AKSARAY', 'NİĞDE', 'KIRIKKALE', 'ÇANKIRI']
results_df = results_df.set_index('city').reindex([c for c in modern_print_order if c in all_cities_list]).reset_index()

print("\n" + "="*125)
print(f"FINAL BOOTSTRAP RESULTS ({N_BOOTSTRAP} runs - Poisson Scaling)")
print("="*125)
table_cols = ['city', 'actual_lat', 'actual_long', 'mean_long', 'mean_lat', 'std_long', 'std_lat', 'error_km']
print(results_df[table_cols].to_string(index=False, float_format="%.3f"))
print(f"\nOverall Average Error: {results_df['error_km'].mean():.2f} km")

# --- Step 6: Individual City Confidence Plots ---
print("\nGenerating Individual Confidence Plots for Modern Cities...")

for name in all_cities_list:
    fig, ax = plt.subplots(figsize=(8, 6))
    load_and_plot_turkey_map(ax)
    
    pts = np.array(boot_results[name])
    act = cities_df.loc[name]
    row_res = results_df[results_df['city'] == name].iloc[0]
    
    # 1. Plot bootstrap points
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=15, color='gray', label='Bootstrap Pts', zorder=3)
    
    # 2. Confidence Ellipses (Ancient Style: Blue 1-Sigma, Red Dashed 2-Sigma)
    draw_confidence_ellipse(pts, ax, n_std=1.0, edgecolor='blue', linewidth=2, label='1-Sigma', zorder=4)
    draw_confidence_ellipse(pts, ax, n_std=2.0, edgecolor='red', linestyle='--', linewidth=1.5, label='2-Sigma', zorder=4)
    
    # 3. Comparison Points
    ax.scatter(row_res['mean_long'], row_res['mean_lat'], c='red', marker='X', s=120, label='MDS Mean', edgecolors='k', zorder=6)
    ax.scatter(act['long_x'], act['lat_y'], c='blue', marker='o', s=100, label='Actual', edgecolors='k', zorder=6)
    
    # 4. Zooming logic (Ancient Style: Pad=0.8)
    pad = 0.8
    all_x = [pts[:, 0].min(), pts[:, 0].max(), act['long_x']]
    all_y = [pts[:, 1].min(), pts[:, 1].max(), act['lat_y']]
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    
    ax.set_title(f"Position Confidence: {name} (Error: {row_res['error_km']:.1f}km)")
    
    # 5. Improved Legend Placement
    ax.legend(loc='upper left', 
              bbox_to_anchor=(1, 1), 
              fontsize='medium', 
              title="Legend", 
              title_fontsize='large',
              framealpha=1)
    
    plt.subplots_adjust(right=0.8) 
    plt.tight_layout()
    plt.show()

# --- Step 7: Summary Comparison Plot (The "Everything" Style) ---
print("\nGenerating Summary Comparison Plot...")
fig, ax = plt.subplots(figsize=(14, 10))
load_and_plot_turkey_map(ax)

# Set Fixed Limits to cover the Central/Eastern Anatolia region used in your data
ax.set_xlim(31, 39.3)
ax.set_ylim(35.8, 42.2)
ax.set_aspect('equal')

# 1. Plot Actual vs MDS Est
ax.scatter(results_df['actual_long'], results_df['actual_lat'], 
           c='blue', marker='o', s=100, label='Actual', edgecolors='k', zorder=5)

ax.scatter(results_df['mean_long'], results_df['mean_lat'], 
           c='red', marker='X', s=100, label='MDS Est.', edgecolors='k', zorder=5)

# 2. Plot Error Lines and Annotations
for index, row in results_df.iterrows():
    # Error line (Red dashed)
    ax.plot([row['actual_long'], row['mean_long']], [row['actual_lat'], row['mean_lat']], 
            'r--', alpha=0.5, zorder=4)
    
    # Label City Name (Blue text, slightly offset)
    ax.text(row['actual_long'], row['actual_lat'] + 0.05, row['city'], 
            fontsize=8, color='blue', fontweight='bold')

ax.set_title(f'Actual vs. MDS Bootstrap Estimates (Modern Cities)\nOverall Mean Error: {results_df["error_km"].mean():.2f} km', fontsize=16)
ax.legend(loc='lower left')

plt.tight_layout()
plt.show()