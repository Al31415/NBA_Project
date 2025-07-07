import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

class DataSplitter:
    def __init__(self, graphs, train_frac=0.8, val_frac=0.1, seed=42, batch_size=128):
        self.graphs = graphs
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = seed
        self.batch_size = batch_size
        
    def split(self):
        idx_all = np.arange(len(self.graphs))
        labels = np.array([g["home_team"].y.item() for g in self.graphs])
        
        # split train vs temp
        can_stratify = len(np.unique(labels)) > 1 and np.min(np.bincount(labels)) >= 2
        train_idx, temp_idx = train_test_split(
            idx_all,
            stratify=labels if can_stratify else None,
            test_size=1 - self.train_frac,
            random_state=self.seed,
        )
        
        # split val vs test
        val_fraction_temp = self.val_frac / (1 - self.train_frac)
        labels_temp = labels[temp_idx]
        can_stratify_temp = (
            len(np.unique(labels_temp)) > 1
            and np.min(np.bincount(labels_temp)) >= 2
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            stratify=labels_temp if can_stratify_temp else None,
            test_size=1 - val_fraction_temp,
            random_state=self.seed,
        )
        
        print({"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)})
        
        train_dl = DataLoader([self.graphs[i] for i in train_idx], batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader([self.graphs[i] for i in val_idx], batch_size=self.batch_size)
        test_dl = DataLoader([self.graphs[i] for i in test_idx], batch_size=self.batch_size)
        
        return train_dl, val_dl, test_dl