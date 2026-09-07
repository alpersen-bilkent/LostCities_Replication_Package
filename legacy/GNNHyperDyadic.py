# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 18:45:00 2026
@author: Gemini
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

# --- Configuration ---
BASE_SEED = 42 
LR_CHOICES = [0.0052, 0.00525, 0.0053, 0.00535]
EMBEDDING_CHOICES = [16]
MAX_EPOCHS = 2100

# --- Data Definition ---
all_city_names = [
    'Durhumit', 'Hahhum', 'Hanaknak', 'Hattus', 'Hurama', 'Kanes', 'Karahna',
    'Kuburnat', 'Malitta', 'Mamma', 'Ninassa', 'Purushaddum', 'Salatuwar',
    'Samuha', 'Sinahuttum', 'Suppiluliya', 'Tapaggas', 'Timelkiya', 'Tuhpiya',
    'Ulama', 'Unipsum', 'Wahsusana', 'Washaniya', 'Zalpa', 'Zimishuna'
]

matrix_data = np.array([
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 13, 5, 0, 1, 0, 0, 1, 2, 0, 0, 10, 0, 0, 1], # Durhumit
    [1, 0, 0, 0, 0, 8, 18, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 19, 0, 0, 1, 4, 0, 0], # Hahhum
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0], # Hanaknak
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], # Hattus
    [0, 2, 0, 0, 0, 15, 1, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0], # Hurama
    [0, 7, 0, 0, 0, 0, 2, 1, 2, 0, 3, 3, 2, 0, 0, 0, 0, 4, 2, 2, 0, 12, 5, 1, 0], # Kanes
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Karahna
    [0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Kuburnat
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], # Malitta
    [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # Mamma
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0], # Ninassa
    [6, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 0], # Purushaddum
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 3, 0, 0, 0, 7, 0, 1, 0], # Salatuwar
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Samuha
    [2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], # Sinahuttum
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Suppiluliya
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Tapaggas
    [0, 3, 1, 0, 21, 19, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0], # Timelkiya
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # Tuhpiya
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # Ulama
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Unipsum
    [5, 0, 0, 1, 0, 3, 0, 0, 0, 0, 1, 22, 19, 0, 0, 0, 1, 1, 7, 0, 0, 0, 0, 1, 0], # Wahsusana
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0], # Washaniya
    [0, 0, 0, 0, 0, 2, 1, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0], # Zalpa
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # Zimishuna
])

cities_df = pd.DataFrame({
    'name': ['Durhumit', 'Hahhum', 'Kuburnat', 'Ninassa', 'Purushaddum', 'Sinahuttum', 'Suppiluliya', 'Tuhpiya', 'Washaniya', 'Zalpa', 'Hattus', 'Kanes', 'Karahna', 'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta', 'Salatuwar', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana', 'Zimishuna', 'Mamma'],
    'lat_y': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 40.021, 38.85, 40.0, 40.148, 40.0, 38.261, 39.363, 39.655, 39.619, 38.027, 38.411, 38.021, 39.584, 40.461, 37.583],
    'long_x': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 34.61, 35.633, 36.1, 35.762, 35.817, 37.114, 33.787, 31.994, 36.528, 38.234, 33.834, 36.503, 33.418, 35.65, 36.933]
})

# --- Prep ---
symmetric_traffic_df = pd.DataFrame(matrix_data, index=all_city_names, columns=all_city_names)
symmetric_traffic_df = symmetric_traffic_df + symmetric_traffic_df.T
known_cities_df = cities_df.dropna().reset_index(drop=True)
NUM_KNOWN_CITIES = len(known_cities_df)
known_city_names = list(known_cities_df['name'])
traffic_matrix_base = symmetric_traffic_df.loc[known_city_names, known_city_names].values
known_city_coords = known_cities_df[['long_x', 'lat_y']].values

scaler = MinMaxScaler()
known_coords_scaled = scaler.fit_transform(known_city_coords)
known_coords_scaled_tensor = torch.tensor(known_coords_scaled, dtype=torch.float)

# --- Classes & Funcs ---
class GCN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, 32)
        self.conv2 = GCNConv(32, 32)
        self.out_head = torch.nn.Linear(32, 2)
    def forward(self, data):
        x = F.relu(self.conv1(self.embedding(data.x), data.edge_index, data.edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.out_head(F.relu(self.conv2(x, data.edge_index, data.edge_attr)))

def create_graph_data(traffic, num_nodes):
    total_volumes = traffic.sum(axis=1)
    total_volumes[total_volumes == 0] = 1.0
    edge_indices, edge_weights = [], []
    rows, cols = np.where(traffic > 0)
    for i, j in zip(rows, cols):
        vol = traffic[i, j]
        edge_indices.append([i, j])
        edge_weights.append(0.5 * ((vol / total_volumes[i]) + (vol / total_volumes[j])))
    return Data(x=torch.arange(num_nodes), edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(edge_weights, dtype=torch.float))

def CustomDistanceLoss(predictions_scaled, targets_scaled, scaler):
    min_val = torch.tensor(scaler.min_, dtype=torch.float)
    scale_val = torch.tensor(scaler.scale_, dtype=torch.float)
    p_un = predictions_scaled / scale_val + min_val
    t_un = targets_scaled / scale_val + min_val
    cos_lat = np.cos(np.radians(37.9))
    d_sq = (t_un[:,1]-p_un[:,1])**2 + (cos_lat*(t_un[:,0]-p_un[:,0]))**2
    return torch.mean((10000/90) * torch.sqrt(d_sq + 1e-6))

# --- MAIN LOGIC: Hyperparameter + Epoch Search ---
grid_history = {} # Stores (lr, emb) -> avg_val_loss_curve (array of length 2100)

print(f"Starting Peak-Potential Grid Search (LOOCV)...")

for emb_dim in EMBEDDING_CHOICES:
    for lr in LR_CHOICES:
        print(f"Testing Config: Emb={emb_dim}, LR={lr}")
        
        # Array to store the average val loss at each epoch across all 15 folds
        agg_val_curve = np.zeros(MAX_EPOCHS)
        
        for i in range(NUM_KNOWN_CITIES):
            # Article Seed: Reset seed for EVERY fold to ensure fair start
            torch.manual_seed(BASE_SEED)
            np.random.seed(BASE_SEED)
            
            train_idx = [j for j in range(NUM_KNOWN_CITIES) if j != i]
            train_mask = torch.tensor(train_idx, dtype=torch.long)
            graph = create_graph_data(traffic_matrix_base, NUM_KNOWN_CITIES)
            
            model = GCN(NUM_KNOWN_CITIES, emb_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            
            for epoch in range(MAX_EPOCHS):
                model.train()
                optimizer.zero_grad()
                out = model(graph)
                loss = CustomDistanceLoss(out[train_mask], known_coords_scaled_tensor[train_mask], scaler)
                loss.backward()
                optimizer.step()
                
                # Validation on the held-out city
                model.eval()
                with torch.no_grad():
                    out_eval = model(graph)
                    val_err = CustomDistanceLoss(out_eval[i:i+1], known_coords_scaled_tensor[i:i+1], scaler)
                    agg_val_curve[epoch] += val_err.item()
        
        # Average across the 15 cities
        agg_val_curve /= NUM_KNOWN_CITIES
        grid_history[(lr, emb_dim)] = agg_val_curve

# --- Analyze Results ---
best_config = None
min_error = float('inf')

heatmap_data = np.zeros((len(EMBEDDING_CHOICES), len(LR_CHOICES)))

for e_idx, emb in enumerate(EMBEDDING_CHOICES):
    for l_idx, lr in enumerate(LR_CHOICES):
        curve = grid_history[(lr, emb)]
        peak_potential = np.min(curve) # The best error this config EVER reached
        heatmap_data[e_idx, l_idx] = peak_potential
        
        if peak_potential < min_error:
            min_error = peak_potential
            best_config = (lr, emb)

best_lr, best_emb = best_config
winning_curve = grid_history[best_config]
best_epoch = np.argmin(winning_curve)

print(f"\nGLOBAL WINNER:")
print(f"LR: {best_lr}, Embedding: {best_emb}")
print(f"Best Epoch: {best_epoch} with Mean Error: {min_error:.2f} km")

# --- Plot 1: Peak Potential Heatmap ---
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, cmap='magma_r', aspect='auto')
plt.colorbar(label='Min LOOCV Error (km)')
plt.xticks(np.arange(len(LR_CHOICES)), LR_CHOICES)
plt.yticks(np.arange(len(EMBEDDING_CHOICES)), EMBEDDING_CHOICES)
plt.xlabel('Learning Rate')
plt.ylabel('Embedding Dimension')
plt.title('Grid Search: Peak Potential (Minimum LOOCV Error per Config)')
# Annotate
for i in range(len(EMBEDDING_CHOICES)):
    for j in range(len(LR_CHOICES)):
        plt.text(j, i, f'{heatmap_data[i, j]:.1f}', ha='center', va='center', color='white' if heatmap_data[i,j] > heatmap_data.mean() else 'black')
plt.tight_layout()
plt.show()

# --- Plot 2: Learning Curve of the Winner ---
# Note: Since we only saved validation curve to save memory/speed, 
# we rerun the winner one last time to get both Train and Val curves for a clean plot
print("\nFinalizing plots for the winning configuration...")
final_train_curve = np.zeros(MAX_EPOCHS)
final_val_curve = winning_curve # We already have the avg val curve

for i in range(NUM_KNOWN_CITIES):
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    train_idx = [j for j in range(NUM_KNOWN_CITIES) if j != i]
    train_mask = torch.tensor(train_idx, dtype=torch.long)
    graph = create_graph_data(traffic_matrix_base, NUM_KNOWN_CITIES)
    model = GCN(NUM_KNOWN_CITIES, best_emb)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=5e-4)
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        t_loss = CustomDistanceLoss(out[train_mask], known_coords_scaled_tensor[train_mask], scaler)
        t_loss.backward()
        optimizer.step()
        final_train_curve[epoch] += t_loss.item()

final_train_curve /= NUM_KNOWN_CITIES

plt.figure(figsize=(10, 6))
plt.plot(final_train_curve, label='Avg Training Loss (LOOCV)', color='blue', alpha=0.5)
plt.plot(final_val_curve, label='Avg Validation Loss (LOOCV)', color='red', linewidth=2)
plt.axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.title(f'Optimal Model Learning Curve\n(LR={best_lr}, Emb={best_emb})')
plt.xlabel('Epochs')
plt.ylabel('Error (km)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()