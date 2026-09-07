# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 13:35:23 2026

@author: user
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

full_traffic_df = pd.DataFrame(matrix_data, index=all_city_names, columns=all_city_names)

cities_data = {
    'name': ['Durhumit', 'Hahhum', 'Kuburnat', 'Ninassa', 'Purushaddum', 'Sinahuttum', 'Suppiluliya', 'Tuhpiya', 'Washaniya', 'Zalpa', 'Hattus', 'Kanes', 'Karahna', 'Tapaggas', 'Hanaknak', 'Hurama', 'Malitta', 'Salatuwar', 'Samuha', 'Timelkiya', 'Ulama', 'Unipsum', 'Wahsusana', 'Zimishuna', 'Mamma'],
    'lat_y': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 40.021, 38.85, 40.0, 40.148, 40.0, 38.261, 39.363, 39.655, 39.619, 38.027, 38.411, 38.021, 39.584, 40.461, 37.583],
    'long_x': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 34.61, 35.633, 36.1, 35.762, 35.817, 37.114, 33.787, 31.994, 36.528, 38.234, 33.834, 36.503, 33.418, 35.65, 36.933]
}

# 1. Symmetricize the trade itineraries
# sym_matrix[i, j] = matrix_data[i, j] + matrix_data[j, i]
sym_matrix = matrix_data + matrix_data.T

# 2. Identify Known vs Unknown Cities based on lat/long availability
known_cities = {name for name, lat in zip(cities_data['name'], cities_data['lat_y']) if not np.isnan(lat)}

# 3. Build the Graph
G = nx.Graph()
for i, city in enumerate(all_city_names):
    status = 'Known' if city in known_cities else 'Unknown'
    G.add_node(city, status=status)

# Strictly add edges ONLY if there is trade (weight > 0)
for i in range(len(all_city_names)):
    for j in range(i + 1, len(all_city_names)):
        weight = sym_matrix[i, j]
        if weight > 0:
            G.add_edge(all_city_names[i], all_city_names[j], weight=weight)

# --- Enhanced Visualization with Edge Weights ---

plt.figure(figsize=(18, 18)) # Increased size for better spacing

# Circular layout ensures no node overlap
pos = nx.circular_layout(G)

# Draw Edges with width scaled by volume
weights = [G[u][v]['weight'] for u, v in G.edges()]
max_w = max(weights)
edge_widths = [0.5 + (w / max_w) * 6 for w in weights]

nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.2, edge_color='gray')

# Draw Nodes
known_nodes = [n for n, d in G.nodes(data=True) if d['status'] == 'Known']
unknown_nodes = [n for n, d in G.nodes(data=True) if d['status'] == 'Unknown']

nx.draw_networkx_nodes(G, pos, nodelist=known_nodes, node_color='#2E86C1', 
                       node_size=1500, alpha=1)
nx.draw_networkx_nodes(G, pos, nodelist=unknown_nodes, node_color='#E67E22', 
                       node_size=1500, alpha=1)

# Draw City Names (slightly outside the circle)
label_pos = {node: (coords[0] * 1.12, coords[1] * 1.12) for node, coords in pos.items()}
nx.draw_networkx_labels(G, label_pos, font_size=11, font_weight='bold')

# --- Add Trade Itinerary Numbers on Edges ---
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(
    G, pos, 
    edge_labels=edge_labels, 
    font_size=9, 
    font_color='darkred',
    label_pos=0.5, # Midpoint of the edge
    rotate=True    # Aligns the number with the line angle
)

plt.title("Trade Network with Itinerary Counts", fontsize=20, pad=30)
plt.axis('off')
plt.show()