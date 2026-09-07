# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 10:36:41 2026

@author: user
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

# --- Configuration ---
FILE_PATH = 'C:/Users/user/OneDrive/Documents/Work/25 Fall/IE490/İller Arası Ticaret.xlsx'
BASE_SEED = 42
MAX_EPOCHS = 2500
LATITUDE_PARAM = 37.9
HIDDEN_CHANNELS = 32
DROPOUT_RATE = 0.2

# Search Space
LR_CHOICES = [0.0048, 0.0049, 0.005, 0.0051]
EMBEDDING_CHOICES = [16]

# --- 1. Data Preparation ---

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
valid_cities = cities_df.index.tolist()
num_cities = len(valid_cities)

try:
    df = pd.read_excel(FILE_PATH, header=0)
    df.rename(columns={df.columns[0]: 'City'}, inplace=True)
    df.set_index('City', inplace=True)
    df = df.replace({'*': 0, '-': 0}).infer_objects(copy=False)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.loc[valid_cities, valid_cities]
    
    # Symmetrize
    raw_vol = df + df.T
    city_totals = raw_vol.sum(axis=1)
    world_total = raw_vol.sum().sum()

    # TII Calculation
    expected = np.outer(city_totals.values, city_totals.values)
    expected[expected == 0] = 1e-9
    tii_matrix = (raw_vol.values * world_total) / expected

    # Log1p and Min-Max Scaling (Article Logic)
    log_tii = np.log1p(tii_matrix)
    t_min, t_max = log_tii.min(), log_tii.max()
    traffic_values = (log_tii - t_min) / (t_max - t_min) if t_max > t_min else log_tii
    np.fill_diagonal(traffic_values, 0)
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    traffic_values = np.zeros((num_cities, num_cities), dtype=np.float32)

scaler = MinMaxScaler()
known_coords_scaled_tensor = torch.tensor(scaler.fit_transform(cities_df[['long_x', 'lat_y']].values), dtype=torch.float)

# --- 2. GNN Components ---

class GCN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, HIDDEN_CHANNELS)
        self.conv2 = GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.out_head = torch.nn.Linear(HIDDEN_CHANNELS, 2)

    def forward(self, data):
        # Forward pass synchronized with Article logic
        x = F.relu(self.conv1(self.embedding(data.x), data.edge_index, data.edge_attr))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training) 
        return self.out_head(F.relu(self.conv2(x, data.edge_index, data.edge_attr)))

def create_graph_data(traffic, num_nodes):
    edge_indices, edge_weights = [], []
    rows, cols = np.where(traffic > 0)
    for i, j in zip(rows, cols):
        edge_indices.append([i, j])
        edge_weights.append(traffic[i, j])
    return Data(x=torch.arange(num_nodes, dtype=torch.long), edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(edge_weights, dtype=torch.float))

def CustomDistanceLoss(predictions_scaled, targets_scaled, scaler):
    min_val = torch.tensor(scaler.min_, dtype=torch.float)
    scale_val = torch.tensor(scaler.scale_, dtype=torch.float)
    p_un = predictions_scaled / scale_val + min_val
    t_un = targets_scaled / scale_val + min_val
    cos_lat = np.cos(np.radians(LATITUDE_PARAM))
    d_sq = (t_un[:,1]-p_un[:,1])**2 + (cos_lat*(t_un[:,0]-p_un[:,0]))**2
    return torch.mean((10000 / 90) * torch.sqrt(d_sq + 1e-6))

# --- 3. Grid Search Logic ---

grid_history = {} 

print(f"Starting Peak-Potential Tuning for Modern Cities (TII - LOOCV)...")

for emb_dim in EMBEDDING_CHOICES:
    for lr in LR_CHOICES:
        print(f"Testing Config: Emb={emb_dim}, LR={lr}")
        agg_val_curve = np.zeros(MAX_EPOCHS)
        
        for i in range(num_cities):
            # Seed Reset per Fold (Crucial for Fairness)
            torch.manual_seed(BASE_SEED)
            np.random.seed(BASE_SEED)
            
            train_idx = [j for j in range(num_cities) if j != i]
            train_mask = torch.tensor(train_idx, dtype=torch.long)
            graph = create_graph_data(traffic_values, num_cities)
            
            model = GCN(num_cities, emb_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            
            for epoch in range(MAX_EPOCHS):
                model.train()
                optimizer.zero_grad()
                out = model(graph)
                loss = CustomDistanceLoss(out[train_mask], known_coords_scaled_tensor[train_mask], scaler)
                loss.backward()
                optimizer.step()
                
                # Validation (Sync PRNG cadence with eval toggle)
                model.eval()
                with torch.no_grad():
                    out_eval = model(graph)
                    val_err = CustomDistanceLoss(out_eval[i:i+1], known_coords_scaled_tensor[i:i+1], scaler)
                    agg_val_curve[epoch] += val_err.item()
        
        agg_val_curve /= num_cities
        grid_history[(lr, emb_dim)] = agg_val_curve

# --- 4. Identify Best Configuration ---

heatmap_data = np.zeros((len(EMBEDDING_CHOICES), len(LR_CHOICES)))
min_error = float('inf')
best_config = None

for e_idx, emb in enumerate(EMBEDDING_CHOICES):
    for l_idx, lr in enumerate(LR_CHOICES):
        curve = grid_history[(lr, emb)]
        peak = np.min(curve)
        heatmap_data[e_idx, l_idx] = peak
        if peak < min_error:
            min_error = peak
            best_config = (lr, emb)

best_lr, best_emb = best_config
best_epoch = np.argmin(grid_history[best_config])

print(f"\nWINNER (TII): LR={best_lr}, Emb={best_emb}, Epoch={best_epoch}, Error={min_error:.2f} km")

# --- 5. Plots ---

# Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap='magma_r', aspect='auto')
plt.colorbar(label='Min LOOCV Error (km)')
plt.xticks(np.arange(len(LR_CHOICES)), LR_CHOICES)
plt.yticks(np.arange(len(EMBEDDING_CHOICES)), EMBEDDING_CHOICES)
plt.xlabel('Learning Rate')
plt.ylabel('Embedding Dimension')
plt.title('Modern Turkey TII Grid Search (Peak Potential)')
for (j, i), val in np.ndenumerate(heatmap_data):
    plt.text(i, j, f'{val:.1f}', ha='center', va='center', color='white' if val > heatmap_data.mean() else 'black')
plt.show()

# Learning Curves for Winner
final_train_curve = np.zeros(MAX_EPOCHS)
for i in range(num_cities):
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    train_idx = [j for j in range(num_cities) if j != i]
    train_mask = torch.tensor(train_idx, dtype=torch.long)
    graph = create_graph_data(traffic_values, num_cities)
    model = GCN(num_cities, best_emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=5e-4)
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        t_loss = CustomDistanceLoss(out[train_mask], known_coords_scaled_tensor[train_mask], scaler)
        t_loss.backward()
        optimizer.step()
        final_train_curve[epoch] += t_loss.item()
final_train_curve /= num_cities

plt.figure(figsize=(10, 6))
plt.plot(final_train_curve, label='Avg Training Loss (LOOCV)', color='blue', alpha=0.5)
plt.plot(grid_history[best_config], label='Avg Validation Loss (LOOCV)', color='red', linewidth=2)
plt.axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.title(f'Modern Turkey TII Learning Curve\n(LR={best_lr}, Emb={best_emb})')
plt.xlabel('Epochs')
plt.ylabel('Error (km)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()