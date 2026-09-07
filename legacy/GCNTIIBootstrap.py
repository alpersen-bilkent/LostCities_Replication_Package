# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 17:07:30 2026

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

# --- Configuration (Synchronized with Dyadic) ---
MASTER_SEED = 42
N_BOOTSTRAP = 40  
SHAPEFILE_PATH = r'C:\Users\user\OneDrive\Documents\Work\25 Fall\IE490\Codes\gadm41_TUR_shp\gadm41_TUR_1.shp'
EPOCHS = 244         
LEARNING_RATE = 0.011 
EMBEDDING_DIM = 8    

# Generate seeds identically to Dyadic code
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

bajramovic_data = {
    'name': ['Hattus', 'Kanes', 'Karahna', 'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta', 'Mamma', 'Salatuwar', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana', 'Zimishuna'],
    'est_lat': [39.997, 39.313, 40.046, 40.0, 40.15, 39.139, 38.888, 38.02, 39.561, 38.3, 38.261, 39.835, 37.583, 39.003, 39.234],
    'est_long': [36.131, 33.918, 34.547, 35.817, 35.761, 38.226, 35.282, 36.503, 33.356, 37.118, 37.114, 33.234, 36.933, 31.926, 34.213]
}

# --- 2. GNN & Helper Logic (Synchronized) ---

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

def calculate_distance_km(p1_long, p1_lat, p2_long, p2_lat):
    cos_factor = np.cos(np.radians(37.9))
    return (10000 / 90) * np.sqrt((p1_lat - p2_lat)**2 + (cos_factor * (p1_long - p2_long))**2)

def load_and_plot_turkey_map(ax):
    try:
        turkey_map = geopandas.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
        turkey_map.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, alpha=0.6, zorder=2)
    except: pass
    try: cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenTopoMap, zorder=0, alpha=0.6)
    except: pass

def create_graph_data(traffic, num_nodes):
    rows, cols = np.where(traffic > 0)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    edge_attr = torch.tensor(traffic[rows, cols], dtype=torch.float)
    return Data(x=torch.arange(num_nodes), edge_index=edge_index, edge_attr=edge_attr)

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

# --- 3. Prep Data ---
cities_df = pd.DataFrame(cities_data)
known_cities_df = cities_df.dropna().reset_index(drop=True)
NUM_KNOWN = len(known_cities_df)
known_names = list(known_cities_df['name'])
known_coords = known_cities_df[['long_x', 'lat_y']].values
actual_coords_dict = dict(zip(known_names, known_coords))

scaler = MinMaxScaler()
scaler.fit(known_coords)
target_scaled_tensor = torch.tensor(scaler.transform(known_coords), dtype=torch.float)

bootstrap_results = {name: [] for name in known_names}

# --- 4. Main Bootstrap Loop (Poisson TII Method) ---
print(f"Executing TII Bootstrapping ({N_BOOTSTRAP} iterations x {NUM_KNOWN} LOO)...")

for seed in tqdm(BOOTSTRAP_SEEDS, desc="Monte Carlo Iterations"):
    # A. Poisson Perturbation
    resampled_raw = np.random.poisson(matrix_data).astype(float)
    
    # B. Recalculate TII (Methodology sync)
    idx = [all_city_names.index(n) for n in known_names]
    sub_raw = resampled_raw[np.ix_(idx, idx)]
    
    x_iW, x_Wj, x_WW = sub_raw.sum(axis=1), sub_raw.sum(axis=0), sub_raw.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        T_ij = np.where(np.outer(x_iW, x_Wj) > 0, (sub_raw * x_WW) / np.outer(x_iW, x_Wj), 0.0)
        T_ji = np.where(np.outer(x_Wj, x_iW) > 0, (sub_raw.T * x_WW) / np.outer(x_Wj, x_iW), 0.0)
    
    sym_tii = 0.5 * (T_ij + T_ji)
    np.fill_diagonal(sym_tii, 0)
    log_tii = np.log1p(sym_tii)
    t_min, t_max = log_tii.min(), log_tii.max()
    norm_tii = (log_tii - t_min) / (t_max - t_min) if (t_max - t_min) > 0 else log_tii
    
    # C. LOOCV Loop
    for i in range(NUM_KNOWN):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        
        train_mask = [j for j in range(NUM_KNOWN) if j != i]
        graph_data = create_graph_data(norm_tii, NUM_KNOWN)
        model = GCN(num_nodes=NUM_KNOWN, embedding_dim=EMBEDDING_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
        
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            out = model(graph_data)
            min_v, scale_v = torch.tensor(scaler.min_, dtype=torch.float), torch.tensor(scaler.scale_, dtype=torch.float)
            p_un, t_un = out[train_mask] / scale_v + min_v, target_scaled_tensor[train_mask] / scale_v + min_v
            cos_l = np.cos(np.radians(37.9))
            d_sq = (t_un[:,1]-p_un[:,1])**2 + (cos_l*(t_un[:,0]-p_un[:,0]))**2
            loss = torch.mean((10000/90) * torch.sqrt(d_sq + 1e-6))
            loss.backward(); optimizer.step()
            
        model.eval()
        with torch.no_grad():
            final_out = model(graph_data)
            pred = scaler.inverse_transform(final_out[i].reshape(1, -1))
            bootstrap_results[known_names[i]].append(pred[0])

# --- 5. Aggregated Plot (The "1") ---
city_print_order = ['Hattus', 'Kanes', 'Karahna', 'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta', 'Mamma', 'Salatuwar', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana', 'Zimishuna']
baj_df = pd.DataFrame(bajramovic_data).set_index('name')

fig, ax = plt.subplots(figsize=(14, 10))
load_and_plot_turkey_map(ax)
ax.set_xlim(31, 39.3); ax.set_ylim(35.8, 42.2); ax.set_aspect('equal')

table_rows = []
for name in city_print_order:
    pts = np.array(bootstrap_results[name])
    mean_pos, std_pos = pts.mean(axis=0), pts.std(axis=0)
    actual, baj = actual_coords_dict[name], baj_df.loc[name]
    err = calculate_distance_km(actual[0], actual[1], mean_pos[0], mean_pos[1])
    
    ax.scatter(actual[0], actual[1], c='blue', marker='o', s=100, label='Actual' if name=='Hattus' else "", edgecolors='k', zorder=5)
    ax.scatter(mean_pos[0], mean_pos[1], c='red', marker='X', s=100, label='GCN Mean' if name=='Hattus' else "", edgecolors='k', zorder=5)
    ax.scatter(baj['est_long'], baj['est_lat'], c='green', marker='s', s=80, label='Bajramovic' if name=='Hattus' else "", edgecolors='k', zorder=4)
    ax.plot([actual[0], mean_pos[0]], [actual[1], mean_pos[1]], 'r--', alpha=0.5)
    ax.plot([actual[0], baj['est_long']], [actual[1], baj['est_lat']], 'g:', alpha=0.6)
    ax.text(actual[0], actual[1] + 0.05, name, fontsize=8, color='blue')
    
    table_rows.append({'City': name, 'Mean_Long': mean_pos[0], 'Mean_Lat': mean_pos[1], 'Std_Long': std_pos[0], 'Std_Lat': std_pos[1], 'Error_KM': err})

ax.set_title(f'Comparison: Actual vs. GCN Means (TII)\nOverall Mean Error: {np.mean([x["Error_KM"] for x in table_rows]):.2f} km', fontsize=16)
ax.legend(loc='lower left'); plt.show()

# --- 6. Individual City Plots (The "15") ---
for name in city_print_order:
    fig, ax = plt.subplots(figsize=(10, 8))
    load_and_plot_turkey_map(ax)
    pts, actual, baj = np.array(bootstrap_results[name]), actual_coords_dict[name], baj_df.loc[name]
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=15, color='gray', label='MC Samples')
    draw_confidence_ellipse(pts, ax, n_std=1.0, edgecolor='blue', label='1-Sigma (68%)')
    draw_confidence_ellipse(pts, ax, n_std=2.0, edgecolor='red', label='2-Sigma (95%)')
    ax.scatter(pts[:,0].mean(), pts[:,1].mean(), c='black', marker='x', s=100, label='GCN Mean', zorder=5)
    ax.scatter(actual[0], actual[1], c='blue', marker='o', s=120, edgecolors='k', label='Actual', zorder=5)
    ax.scatter(baj['est_long'], baj['est_lat'], c='orange', marker='s', s=80, edgecolors='k', label='Barjamovic', zorder=4)
    pad = 0.8
    ax.set_xlim(min(pts[:,0].min(), actual[0], baj['est_long']) - pad, max(pts[:,0].max(), actual[0], baj['est_long']) + pad)
    ax.set_ylim(min(pts[:,1].min(), actual[1], baj['est_lat']) - pad, max(pts[:,1].max(), actual[1], baj['est_lat']) + pad)
    ax.set_title(f"Uncertainty Quantification: {name}\n(TII Model - N=40 Runs)"); ax.legend(loc='lower left'); plt.tight_layout(); plt.show()

# --- Final Synchronized Table ---
results_final_df = pd.DataFrame(table_rows)
print("\n" + "="*80 + f"\nBOOTSTRAPPED VALIDATION RESULTS (TII MODEL - N={N_BOOTSTRAP})\n" + "="*80)
print(results_final_df.round(3).to_string(index=False))
print("="*80)