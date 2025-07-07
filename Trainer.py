import torch
import torch.nn.functional as F
import copy
import math
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-3
    edge_dropout_p: float = 0.1
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    max_epochs: int = 100
    patience: int = 10
    device: Optional[torch.device] = None
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
#'lr': 0.001, 'wd': 0.001, 'hid': 40, 'dropout': 0.2

class Trainer:
    def __init__(self, model, head, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        if self.config.device is None:
            self.config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model = model
        self.head = head
        self.device = self.config.device
        self.opt = torch.optim.Adam(
            list(model.parameters()) + list(head.parameters()),
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, 
            mode="min", 
            factor=self.config.scheduler_factor, 
            patience=self.config.scheduler_patience, 
            verbose=True
        )
        
    def edge_dropout(self, batch: HeteroData) -> HeteroData:
        b = copy.deepcopy(batch)
        for rel in b.edge_index_dict.keys():
            ei = b[rel].edge_index
            ei2, mask = dropout_edge(ei, p=self.config.edge_dropout_p, training=True)
            b[rel].edge_index = ei2.long()
            if "edge_attr" in b[rel]:
                b[rel].edge_attr = b[rel].edge_attr[mask]
        return b
        
    def smooth_ce(self, logits, target, smoothing=None):
        smoothing = smoothing or self.config.label_smoothing
        n = logits.size(1)
        with torch.no_grad():
            dist = torch.zeros_like(logits).fill_(smoothing / (n - 1))
            dist.scatter_(1, target.unsqueeze(1), 1 - smoothing)
        return -(dist * F.log_softmax(logits, 1)).sum(1).mean()
        
    def train_epoch(self, train_dl):
        self.model.train()
        self.head.train()
        tl = tc = ts = 0
        for batch in train_dl:
            batch = self.edge_dropout(batch.to(self.device))
            out = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            y = batch["home_team"].y.view(-1)
            logits = self.head(torch.cat([
                out["home_team"],
                out["visitor_team"],
                out["home_team"] - out["visitor_team"]
            ], 1))
            loss = self.smooth_ce(logits, y)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.opt.step()
            tl += loss.item() * y.size(0)
            tc += (logits.argmax(1) == y).sum().item()
            ts += y.size(0)
        return tl / ts, tc / ts
        
    def validate(self, val_dl):
        self.model.eval()
        self.head.eval()
        vl = vc = vs = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = batch.to(self.device)
                out = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                y = batch["home_team"].y.view(-1)
                logits = self.head(torch.cat([
                    out["home_team"],
                    out["visitor_team"],
                    out["home_team"] - out["visitor_team"]
                ], 1))
                loss = self.smooth_ce(logits, y)
                vl += loss.item() * y.size(0)
                vc += (logits.argmax(1) == y).sum().item()
                vs += y.size(0)
        return vl / vs, vc / vs
    def train(self, train_dl, val_dl):
        best_vl = math.inf
        bad = 0
        for epoch in range(1, self.config.max_epochs + 1):
            tr_loss, tr_acc = self.train_epoch(train_dl)
            vl_loss, vl_acc = self.validate(val_dl)
            self.scheduler.step(vl_loss)
            
            print(f"Ep {epoch:02d} | train {tr_loss:.3f}/{tr_acc:.2%} | "
                  f"val {vl_loss:.3f}/{vl_acc:.2%}")
                  
            if vl_loss < best_vl - 1e-4:
                best_vl, bad = vl_loss, 0
                self.save_model("best.pt")
            else:
                bad += 1
                if bad >= self.config.patience:
                    print(f"Early stop @ epoch {epoch}")
                    break
        
    def test(self, test_dl):
        ckpt = torch.load("best.pt", map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.head.load_state_dict(ckpt["head"])
        self.model.eval()
        self.head.eval()
        tl = tc = ts = 0
        with torch.no_grad():
            for batch in test_dl:
                batch = batch.to(self.device)
                out = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                y = batch["home_team"].y.view(-1)
                logits = self.head(torch.cat([
                    out["home_team"],
                    out["visitor_team"],
                    out["home_team"] - out["visitor_team"]
                ], 1))
                loss = self.smooth_ce(logits, y)
                tl += loss.item() * y.size(0)
                tc += (logits.argmax(1) == y).sum().item()
                ts += y.size(0)
        print(f"TEST  loss {tl/ts:.3f} | acc {tc/ts:.2%}")
        
    def save_model(self, path="model.pt"):
        torch.save({
            "model": self.model.state_dict(),
            "head": self.head.state_dict()
        }, path)
        
    def load_model(self, path="model.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.head.load_state_dict(ckpt["head"])
        
 