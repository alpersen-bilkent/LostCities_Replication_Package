# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 00:54:15 2026

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
import geopandas
import contextily as cx
from tqdm import tqdm

# --- Configuration ---
MASTER_SEED = 42
N_BOOTSTRAP = 200  
SHAPEFILE_PATH = 'C:\\Users\\user\\OneDrive\\Documents\\Work\\25 Fall\\IE490\\Codes\\gadm41_TUR_shp\\gadm41_TUR_1.shp' 
EPOCHS = 1473         
LEARNING_RATE = 0.0053 
EMBEDDING_DIM = 16    

np.random.seed(MASTER_SEED)
BOOTSTRAP_SEEDS = np.random.randint(0, 10000, size=N_BOOTSTRAP)

# --- 1. Data Definition ---
all_city_names = [
    'Durhumit', 'Hahhum', 'Hanaknak', 'Hattus', 'Hurama', 'Kanes', 'Karahna',
    'Kuburnat', 'Malitta', 'Mamma', 'Ninassa', 'Purushaddum', 'Salatuwar',
    'Samuha', 'Sinahuttum', 'Suppiluliya', 'Tapaggas', 'Timelkiya', 'Tuhpiya',
    'Ulama', 'Unipsum', 'Wahsusana', 'Washaniya', 'Zalpa', 'Zimishuna'
]

matrix_data = np.array([
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

cities_data = {
    'name': ['Durhumit', 'Hahhum', 'Kuburnat', 'Ninassa', 'Purushaddum', 'Sinahuttum', 'Suppiluliya', 'Tuhpiya', 'Washaniya', 'Zalpa', 'Hattus', 'Kanes', 'Karahna', 'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta', 'Salatuwar', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana', 'Zimishuna', 'Mamma'],
    'lat_y': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 40.021, 38.85, 40.0, 40.148, 40.0, 38.261, 39.363, 39.655, 39.619, 38.027, 38.411, 38.021, 39.584, 40.461, 37.583],
    'long_x': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 34.61, 35.633, 36.1, 35.762, 35.817, 37.114, 33.787, 31.994, 36.528, 38.234, 33.834, 36.503, 33.418, 35.65, 36.933]
}

bajramovic_lost_data = {
    'name': ['Durhumit', 'Hahhum', 'Kuburnat', 'Ninassa', 'Purushaddum', 'Sinahuttum', 'Suppiluliya', 'Tuhpiya', 'Washaniya', 'Zalpa'],
    'est_lat': [40.47, 38.43, 40.71, 38.98, 39.71, 39.96, 40.02, 39.61, 39.16, 38.81],
    'est_long': [35.65, 38.04, 36.52, 34.61, 32.87, 34.87, 34.62, 35.2, 34.31, 37.86]
}

# --- 2. Helper Functions ---

class GCN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, 32)
        self.conv2 = GCNConv(32, 32)
        self.out_head = torch.nn.Linear(32, 2)
    def forward(self, data):
        x = self.embedding(data.x)
        x = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        return self.out_head(x)

def calculate_gravity_weights(traffic_matrix):
    # Structural Gravity: Integration Score (I_ij)
    M_j = traffic_matrix.sum(axis=0)
    M_j[M_j == 0] = 1e-9
    S_matrix = traffic_matrix / M_j
    I_matrix = 0.5 * (S_matrix + S_matrix.T)
    return I_matrix

def create_graph_data(integration_matrix, num_nodes):
    edge_indices, edge_weights = [], []
    rows, cols = np.where(integration_matrix > 0)
    for i, j in zip(rows, cols):
        edge_indices.append([i, j])
        edge_weights.append(integration_matrix[i, j])
    return Data(x=torch.arange(num_nodes), 
                edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), 
                edge_attr=torch.tensor(edge_weights, dtype=torch.float))

def calculate_distance_km(p1_long, p1_lat, p2_long, p2_lat):
    cos_factor = np.cos(np.radians(37.9))
    y_diff = p1_lat - p2_lat
    x_diff = p1_long - p2_long
    return (10000 / 90) * np.sqrt(y_diff**2 + (cos_factor * x_diff)**2)

def load_and_plot_turkey_map(ax):
    try:
        turkey_map = geopandas.read_file(SHAPEFILE_PATH)
        turkey_map = turkey_map.to_crs(epsg=4326)
        turkey_map.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, alpha=0.6, zorder=2)
    except: pass
    try: cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenTopoMap, zorder=0, alpha=0.6)
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

# --- 3. Preparation ---
cities_df = pd.DataFrame(cities_data)
known_mask = cities_df['lat_y'].notna()
known_indices = np.where(known_mask)[0]
lost_indices = np.where(~known_mask)[0]
lost_names = cities_df.iloc[lost_indices]['name'].tolist()

known_coords = cities_df.iloc[known_indices][['long_x', 'lat_y']].values
scaler = MinMaxScaler().fit(known_coords)

baj_df = pd.DataFrame(bajramovic_lost_data).set_index('name')
bootstrap_results = {name: [] for name in lost_names}

# --- 4. Bootstrap Training Loop ---
print(f"Sampling Predictions for {len(lost_names)} lost cities (N={N_BOOTSTRAP})...")

for seed in tqdm(BOOTSTRAP_SEEDS):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    
    # Resample trade and calculate Gravity weights
    resampled_traffic = np.random.poisson(matrix_data).astype(float)
    i_matrix = calculate_gravity_weights(resampled_traffic)
    graph_data = create_graph_data(i_matrix, len(all_city_names))
    
    model = GCN(num_nodes=len(all_city_names), embedding_dim=EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    target_scaled = torch.tensor(scaler.transform(known_coords), dtype=torch.float)
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = model(graph_data)
        
        # Loss calculated on known cities only
        p_un = out[known_indices] / torch.tensor(scaler.scale_) + torch.tensor(scaler.min_)
        t_un = target_scaled / torch.tensor(scaler.scale_) + torch.tensor(scaler.min_)
        cos_l = np.cos(np.radians(37.9))
        d_sq = (t_un[:,1]-p_un[:,1])**2 + (cos_l*(t_un[:,0]-p_un[:,0]))**2
        loss = torch.mean((10000/90) * torch.sqrt(d_sq + 1e-6))
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        final_out = model(graph_data)
        preds = scaler.inverse_transform(final_out[lost_indices].numpy())
        for i, name in enumerate(lost_names):
            bootstrap_results[name].append(preds[i])

# --- 5. Output Generation ---

# A. Table
table_rows = []
for name in lost_names:
    pts = np.array(bootstrap_results[name])
    mean_c = np.mean(pts, axis=0)
    std_c = np.std(pts, axis=0)
    baj = baj_df.loc[name]
    dist = calculate_distance_km(mean_c[0], mean_c[1], baj['est_long'], baj['est_lat'])
    table_rows.append({'City': name, 'Mean_Long': mean_c[0], 'Mean_Lat': mean_c[1], 
                        'Std_Long': std_c[0], 'Std_Lat': std_c[1], 'Dist_to_Baj': dist})

results_df = pd.DataFrame(table_rows)
print("\n--- PREDICTION RESULTS ---")
print(results_df.round(3).to_string(index=False))

# B. Individual Confidence Plots
for name in lost_names:
    fig, ax = plt.subplots(figsize=(8, 6))
    load_and_plot_turkey_map(ax)
    pts = np.array(bootstrap_results[name])
    baj = baj_df.loc[name]
    
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=15, color='gray')
    draw_confidence_ellipse(pts, ax, n_std=1.0, edgecolor='blue', label='1-Sigma')
    draw_confidence_ellipse(pts, ax, n_std=2.0, edgecolor='red', label='2-Sigma')
    ax.scatter(np.mean(pts[:,0]), np.mean(pts[:,1]), c='red', marker='X', s=100, label='GCN Mean')
    ax.scatter(baj['est_long'], baj['est_lat'], c='green', marker='s', s=80, label='Barjamovic')
    
    pad = 0.7
    ax.set_xlim(min(pts[:,0].min(), baj['est_long']) - pad, max(pts[:,0].max(), baj['est_long']) + pad)
    ax.set_ylim(min(pts[:,1].min(), baj['est_lat']) - pad, max(pts[:,1].max(), baj['est_lat']) + pad)
    ax.set_title(f"Prediction Confidence: {name}")
    ax.legend()
    plt.show()

# C. Overall Prediction Map
fig, ax = plt.subplots(figsize=(12, 10))
load_and_plot_turkey_map(ax)
ax.set_xlim(31, 39.3); ax.set_ylim(35.8, 42.2);  ax.set_aspect('equal')

for name in lost_names:
    pts = np.array(bootstrap_results[name])
    mean_c = np.mean(pts, axis=0)
    baj = baj_df.loc[name]
    
    ax.scatter(mean_c[0], mean_c[1], c='red', marker='X', s=100, zorder=5)
    ax.scatter(baj['est_long'], baj['est_lat'], c='green', marker='s', s=80, zorder=5)
    ax.plot([mean_c[0], baj['est_long']], [mean_c[1], baj['est_lat']], 'k--', alpha=0.4)
    ax.text(mean_c[0], mean_c[1] + 0.05, name, fontsize=8, color='red', fontweight='bold')

ax.set_title(f"Final GCN Predictions vs Barjamovic\nAvg Distance: {results_df['Dist_to_Baj'].mean():.2f} km")
plt.show()