# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 23:57:01 2026

@author: user
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
import contextily as cx
from tqdm import tqdm
import os

# --- Configuration ---
MASTER_SEED = 42
N_BOOTSTRAP = 40  
EPOCHS = 596
LEARNING_RATE = 0.005
EMBEDDING_DIM = 16     
HIDDEN_CHANNELS = 32
DROPOUT_RATE = 0.2    
LATITUDE_PARAM = 37.9 

SHAPEFILE_PATH = r'C:\Users\user\OneDrive\Documents\Work\25 Fall\IE490\Codes\gadm41_TUR_shp\gadm41_TUR_1.shp' 
FILE_PATH = 'C:/Users/user/OneDrive/Documents/Work/25 Fall/IE490/İller Arası Ticaret.xlsx'

# Generate seeds
np.random.seed(MASTER_SEED)
BOOTSTRAP_SEEDS = np.random.randint(0, 10000, size=N_BOOTSTRAP)

# --- 1. Helper Functions ---

def calculate_distance_km(p1_long, p1_lat, p2_long, p2_lat):
    cos_factor = np.cos(np.radians(LATITUDE_PARAM))
    y_diff = p1_lat - p2_lat
    x_diff = p1_long - p2_long
    return (10000 / 90) * np.sqrt(y_diff**2 + (cos_factor * x_diff)**2)

def load_and_plot_turkey_map(ax):
    if os.path.exists(SHAPEFILE_PATH):
        try:
            turkey_map = gpd.read_file(SHAPEFILE_PATH)
            turkey_map = turkey_map.to_crs(epsg=4326)
            turkey_map.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, alpha=0.6, zorder=2)
        except: pass
    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenTopoMap, zorder=0, alpha=0.6)
    except: pass

def draw_confidence_ellipse(data, ax, n_std=1.0, facecolor='none', **kwargs):
    cov = np.cov(data, rowvar=False)
    mean = np.mean(data, axis=0)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, facecolor=facecolor, **kwargs)
    return ax.add_patch(ellipse)

def create_graph_data(traffic, num_nodes):
    edge_indices, edge_weights = [], []
    rows, cols = np.where(traffic > 0)
    for i, j in zip(rows, cols):
        edge_indices.append([i, j])
        edge_weights.append(traffic[i, j])
    return Data(x=torch.arange(num_nodes), 
                edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), 
                edge_attr=torch.tensor(edge_weights, dtype=torch.float))

class GCN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, HIDDEN_CHANNELS)
        self.conv2 = GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.out_head = torch.nn.Linear(HIDDEN_CHANNELS, 2)
    def forward(self, data):
        x = self.embedding(data.x)
        x = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        return self.out_head(x)

# --- 2. Data Loading & Prep ---
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
valid_cities = sorted(cities_df.index.tolist())
num_cities = len(valid_cities)

df_raw = pd.read_excel(FILE_PATH, header=0)
df_raw.rename(columns={df_raw.columns[0]: 'City'}, inplace=True)
df_raw.set_index('City', inplace=True)
df_raw = df_raw.replace({'*': 0, '-': 0}).infer_objects(copy=False)
df_raw = df_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
df_subset = df_raw.loc[valid_cities, valid_cities]
symmetric_raw_matrix = (df_subset + df_subset.T).values.astype(np.float32)

known_coords = cities_df.loc[valid_cities, ['long_x', 'lat_y']].values
scaler = MinMaxScaler()
scaler.fit(known_coords)

bootstrap_results = {name: [] for name in valid_cities}

# --- 3. Main Bootstrap Loop ---
print(f"Executing TII Monte Carlo Sampling ({N_BOOTSTRAP} seeds x {num_cities} LOO)...")

# Poisson scaling factor
max_val = symmetric_raw_matrix.max()
scaling_factor = 1e3 / max_val if max_val > 1e3 else 1.0
scaled_base_matrix = symmetric_raw_matrix * scaling_factor

for seed_idx, seed in enumerate(tqdm(BOOTSTRAP_SEEDS, desc="Monte Carlo Iterations")):
    # A. Resample Trade
    resampled_trade = np.random.poisson(scaled_base_matrix).astype(float)
    
    # B. Calculate TII for this bootstrap instance
    city_totals = resampled_trade.sum(axis=1)
    world_total = resampled_trade.sum()
    expected = np.outer(city_totals, city_totals)
    expected[expected == 0] = 1e-9
    tii_inst = (resampled_trade * world_total) / expected
    
    # Log transform & Scale to [0, 1]
    log_tii = np.log1p(tii_inst)
    t_min, t_max = log_tii.min(), log_tii.max()
    traffic_values = (log_tii - t_min) / (t_max - t_min) if t_max > t_min else log_tii
    np.fill_diagonal(traffic_values, 0)

    # C. LOO Loop
    for i in range(num_cities):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        
        train_mask = [j for j in range(num_cities) if j != i]
        graph_data = create_graph_data(traffic_values, num_cities)
        
        model = GCN(num_nodes=num_cities, embedding_dim=EMBEDDING_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
        target_scaled = torch.tensor(scaler.transform(known_coords), dtype=torch.float)
        
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            out = model(graph_data)
            
            min_v = torch.tensor(scaler.min_, dtype=torch.float)
            scale_v = torch.tensor(scaler.scale_, dtype=torch.float)
            p_un = out[train_mask] / scale_v + min_v
            t_un = target_scaled[train_mask] / scale_v + min_v
            
            cos_l = np.cos(np.radians(LATITUDE_PARAM))
            d_sq = (t_un[:,1]-p_un[:,1])**2 + (cos_l*(t_un[:,0]-p_un[:,0]))**2
            loss = torch.mean((10000/90) * torch.sqrt(d_sq + 1e-6))
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            final_out = model(graph_data)
            pred = scaler.inverse_transform(final_out[i].reshape(1, -1))
            bootstrap_results[valid_cities[i]].append(pred[0])

# --- 4. Statistics and Plotting ---
table_rows = []
for name in valid_cities:
    pts = np.array(bootstrap_results[name])
    mean_coord = np.mean(pts, axis=0)
    std_coord = np.std(pts, axis=0)
    actual = known_coords[valid_cities.index(name)]
    err = calculate_distance_km(actual[0], actual[1], mean_coord[0], mean_coord[1])
    table_rows.append({
        'City': name,
        'Mean_Long': mean_coord[0], 'Mean_Lat': mean_coord[1],
        'Std_Long': std_coord[0], 'Std_Lat': std_coord[1],
        'Error_KM': err
    })

results_final_df = pd.DataFrame(table_rows)

# Individual Confidence Plots
for name in valid_cities:
    fig, ax = plt.subplots(figsize=(10, 8))
    load_and_plot_turkey_map(ax)
    pts = np.array(bootstrap_results[name])
    actual = known_coords[valid_cities.index(name)]
    
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=15, color='gray', label='MC Samples')
    draw_confidence_ellipse(pts, ax, n_std=1.0, edgecolor='blue', label='1-Sigma (68%)')
    draw_confidence_ellipse(pts, ax, n_std=2.0, edgecolor='red', label='2-Sigma (95%)')
    
    ax.scatter(np.mean(pts[:,0]), np.mean(pts[:,1]), c='black', marker='x', s=100, label='GCN Mean', zorder=5)
    ax.scatter(actual[0], actual[1], c='blue', marker='o', s=120, edgecolors='k', label='Actual', zorder=5)
    
    pad = 0.6
    ax.set_xlim(min(pts[:,0].min(), actual[0]) - pad, max(pts[:,0].max(), actual[0]) + pad)
    ax.set_ylim(min(pts[:,1].min(), actual[1]) - pad, max(pts[:,1].max(), actual[1]) + pad)
    ax.set_title(f"Uncertainty Quantification (TII): {name}\n(N={N_BOOTSTRAP} Bootstrap Runs)")
    ax.legend(loc='lower left')
    plt.tight_layout(); plt.show()

# Final Aggregate Map
fig, ax = plt.subplots(figsize=(14, 10))
load_and_plot_turkey_map(ax)
ax.set_xlim(31, 39.3); ax.set_ylim(35.8, 42.2); ax.set_aspect('equal')

for name in valid_cities:
    mean_c = results_final_df.loc[results_final_df['City'] == name, ['Mean_Long', 'Mean_Lat']].values[0]
    act = known_coords[valid_cities.index(name)]
    ax.scatter(act[0], act[1], c='blue', marker='o', s=80, edgecolors='k', zorder=5)
    ax.scatter(mean_c[0], mean_c[1], c='red', marker='X', s=80, edgecolors='k', zorder=5)
    ax.plot([act[0], mean_c[0]], [act[1], mean_c[1]], 'r--', alpha=0.4)
    ax.text(act[0], act[1] + 0.03, name, fontsize=8, color='blue', ha='center')

ax.set_title(f'Modern Turkey GCN (TII): Bootstrapped Means\nOverall Mean Error: {results_final_df["Error_KM"].mean():.2f} km', fontsize=16)
plt.show()

print("\n--- TII BOOTSTRAPPED VALIDATION RESULTS ---")
print(results_final_df.round(3).to_string(index=False))