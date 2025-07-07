from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GENConv, to_hetero, LayerNorm
from Graph_Creation import GraphCreator
from DataSplitter import DataSplitter
from Trainer import Trainer

# Data parameters
DATA_CONFIG = {
    'csv_path': Path("data/nba_regular_season_playbyplay_data.csv"),
    'team_ids_csv': Path("data/team_ids.csv"),
    'debug_n': 0,  # 0 = full dataset
    'first_half_only': True,
    'edge_dim': 5,
    'node_feats': 6,
}

# Model hyperparameters 
MODEL_CONFIG = {
    'hidden_dim': 40,
    'dropout': 0.2
}

class HomoBackbone(nn.Module):
    """Homogeneous GNN backbone that will be converted to heterogeneous."""
    def __init__(self, node_feats: int, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.conv = GENConv(
            node_feats, 
            hidden_dim,
            aggr="softmax",
            t=1.0, 
            learn_t=True,
            edge_dim=edge_dim
        )
        self.norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.conv(x, edge_index, edge_attr=edge_attr))
        return self.norm(self.dropout(h))

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and process graphs
    graph_creator = GraphCreator(
        csv_path=DATA_CONFIG['csv_path'],
        team_ids_csv_path=DATA_CONFIG['team_ids_csv'],
        debug_n=DATA_CONFIG['debug_n'],
        first_half_only=DATA_CONFIG['first_half_only'],
        edge_dim=DATA_CONFIG['edge_dim'],
        node_feats=DATA_CONFIG['node_feats']
    )
    graphs = graph_creator.create_graphs()

    # Split data
    data_splitter = DataSplitter(graphs)
    train_data, val_data, test_data = data_splitter.split()

    # Initialize model
    backbone = HomoBackbone(
        node_feats=DATA_CONFIG['node_feats'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        edge_dim=DATA_CONFIG['edge_dim'],
        dropout=MODEL_CONFIG['dropout']
    )
    model = to_hetero(backbone, metadata=graphs[0].metadata()).to(device)
    head = nn.Linear(MODEL_CONFIG['hidden_dim'] * 3, 2).to(device)

    # Train model
    trainer = Trainer(model, head)
    trainer.train(train_data, val_data)

if __name__ == "__main__":
    main()
