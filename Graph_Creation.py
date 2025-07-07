from __future__ import annotations
from pathlib import Path
import collections
import numpy as np, pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import HeteroData

class GraphCreator:
    def __init__(self, csv_path: str, team_ids_csv_path: str, debug_n: int = 0, first_half_only: bool = True,
                 edge_dim: int = 5, node_feats: int = 6):
        # ─── hyper-parameters ───────────────────────────────────────
        self.CSV = Path(csv_path)
        self.TEAM_IDS_CSV = Path(team_ids_csv_path)
        self.DEBUG_N = debug_n          # 0 ⇒ full dataset
        self.FIRST_HALF_ONLY = first_half_only
        self.EDGE_DIM = edge_dim
        self.NODE_FEATS = node_feats

        # ─── load CSV & event→channel map ───────────────────────────
        self.df = pd.read_csv(self.CSV, low_memory=False)
        
        #From the description of the event, identify the action corresponding to each play 
        self.ACTIONS = ["assist", "block", "steal", "foul", "rebound"]
        self.event2action = {}
        for (m, a), g in self.df.groupby(["EVENTMSGTYPE", "EVENTMSGACTIONTYPE"]):
            txt = "".join(
                s for s in pd.unique(
                    g[["HOMEDESCRIPTION","VISITORDESCRIPTION","NEUTRALDESCRIPTION"]]
                    .values.ravel()) if isinstance(s, str)
            ).lower()
            if m == 1 and "ast"  in txt: self.event2action[(m,a)] = "assist"
            elif m == 2 and "blk" in txt: self.event2action[(m,a)] = "block"
            elif m == 5 and "steal" in txt: self.event2action[(m,a)] = "steal"
            elif m == 6:                  self.event2action[(m,a)] = "foul"
            elif m == 4:                  self.event2action[(m,a)] = "rebound"

        self.team_ids = pd.read_csv(self.TEAM_IDS_CSV)
        self.gid2tids = {gid: self.get_team_ids(gid) for gid, g in self.df.groupby("GAME_ID")}

        # Constants
        self.TEAM_COLS = ["PLAYER1_TEAM_ID","PLAYER2_TEAM_ID","PLAYER3_TEAM_ID"]
        self.PID_COLS = ["PLAYER1_ID","PLAYER2_ID","PLAYER3_ID"]
        self.ZERO_VEC = lambda: torch.zeros(self.EDGE_DIM)
        self.ALL_EDGE_TYPES = [
            ("home_player",    "member_of", "home_team"),
            ("visitor_player", "member_of", "visitor_team"),
            ("home_player",    "interact",  "home_player"),
            ("home_player",    "interact",  "visitor_player"),
            ("visitor_player", "interact",  "home_player"),
            ("visitor_player", "interact",  "visitor_player"),
            ("home_team",      "versus",    "visitor_team"),
            ("visitor_team",   "versus",    "home_team"),
        ]

    def is_three(self, r):
        return r.EVENTMSGTYPE == 1 and any(
            isinstance(r[c], str) and "3pt" in r[c].lower()
            for c in ("HOMEDESCRIPTION","VISITORDESCRIPTION","NEUTRALDESCRIPTION"))

    def get_team_ids(self, game_id):
        game = self.team_ids.loc[self.team_ids.GAME_ID == game_id]
        home_team_id = game.HOME_TEAM_ID.iloc[0]
        visitor_team_id = game.VISITOR_TEAM_ID.iloc[0]
        return home_team_id, visitor_team_id

    def build_graph(self, full_gdf: pd.DataFrame) -> HeteroData:
        #Get the home and visitor team ids for the game
        gid = full_gdf.GAME_ID.iat[0]
        home_tid, vis_tid = self.gid2tids[gid]

        #Get the score for the game
        vis_pts, home_pts = map(int, full_gdf.SCORE.dropna().iloc[-1]
                                .replace(" ", "").split("-"))
        #Set the label for the game to be 1 if the home team won, 0 otherwise
        label = torch.tensor([home_pts > vis_pts], dtype=torch.long)

        #Get the first half of the game as the input data for the graph
        gdf = full_gdf[full_gdf.PERIOD <= 2].copy() if self.FIRST_HALF_ONLY else full_gdf

        # rosters
        #Initialize the home and visitor player sets
        home_p, vis_p = set(), set()
        #Get the players for the home and visitor teams by looking at the player ids and team ids
        for pcol, tcol in zip(self.PID_COLS, self.TEAM_COLS):
            sub = gdf[[pcol, tcol]].dropna()
            home_p |= set(sub.loc[sub[tcol] == home_tid, pcol].astype(int))
            vis_p  |= set(sub.loc[sub[tcol] == vis_tid,  pcol].astype(int))
        home_p, vis_p = sorted(home_p), sorted(vis_p)
        #Create a mapping of player ids to indices for the home and visitor player sets
        hp2i, vp2i = {p: i for i, p in enumerate(home_p)}, {p: i for i, p in enumerate(vis_p)}

        def feats(pid):
            #Get the number of 2-pointers, 3-pointers, free throws, assists, rebounds, and turnovers for a given player
            fg  = gdf[(gdf.EVENTMSGTYPE == 1) & (gdf.PLAYER1_ID == pid)]
            n3  = fg.apply(self.is_three, axis=1).sum(); n2 = len(fg) - n3
            ft  = gdf[(gdf.EVENTMSGTYPE == 3) & (gdf.PLAYER1_ID == pid)].SCORE.notna().sum()
            ast = (gdf.EVENTMSGTYPE.eq(1) & gdf.PLAYER2_ID.eq(pid)).sum()
            reb = (gdf.EVENTMSGTYPE.eq(4) & gdf.PLAYER1_ID.eq(pid)).sum()
            tov = (gdf.EVENTMSGTYPE.eq(5) & gdf.PLAYER1_ID.eq(pid)).sum()
            return torch.tensor([n2, n3, ft, ast, reb, tov], dtype=torch.float)

        #Get the features for the home and visitor players
        x_home = torch.stack([feats(p) for p in home_p]) if home_p else torch.empty((0, self.NODE_FEATS))
        x_vis  = torch.stack([feats(p) for p in vis_p ]) if vis_p  else torch.empty((0, self.NODE_FEATS))

        #Create the graph data object
        data = HeteroData()
        data["home_player"].x    = x_home
        data["visitor_player"].x = x_vis
        data["home_team"].x      = x_home.sum(0, True) if len(x_home) else torch.zeros((1, self.NODE_FEATS))
        data["visitor_team"].x   = x_vis.sum(0, True) if len(x_vis) else torch.zeros((1, self.NODE_FEATS))

        # member_of edges
        if home_p:
            idx = torch.arange(len(home_p), dtype=torch.long)
            et = ("home_player", "member_of", "home_team")
            data[et].edge_index = torch.stack([idx, torch.zeros_like(idx)])
            data[et].edge_attr  = torch.zeros((len(home_p), self.EDGE_DIM))
        if vis_p:
            idx = torch.arange(len(vis_p), dtype=torch.long)
            et = ("visitor_player", "member_of", "visitor_team")
            data[et].edge_index = torch.stack([idx, torch.zeros_like(idx)])
            data[et].edge_attr  = torch.zeros((len(vis_p), self.EDGE_DIM))

        # interaction edges
        inter = collections.defaultdict(self.ZERO_VEC)
        last = None
        for _, r in gdf.iterrows():
            action = self.event2action.get((r.EVENTMSGTYPE, r.EVENTMSGACTIONTYPE))
            p1 = int(r.PLAYER1_ID) if pd.notna(r.PLAYER1_ID) else None
            p2 = int(r.PLAYER2_ID) if pd.notna(r.PLAYER2_ID) else None
            p3 = int(r.PLAYER3_ID) if pd.notna(r.PLAYER3_ID) else None
            if action == "assist" and p1 and p2: inter[(p2, p1)][0] += 1; last = p1
            elif action == "block"  and p1 and p3: inter[(p3, p1)][1] += 1; last = p1
            elif action == "steal"  and p1 and p2: inter[(p2, p1)][2] += 1; last = None
            elif action == "foul"   and p1 and p2: inter[(p1, p2)][3] += 1
            elif action == "rebound" and p1 and last: inter[(p1, last)][4] += 1; last = None
            if r.EVENTMSGTYPE in (1, 2) and p1: last = p1

        for (sid, did), vec in inter.items():
            if vec.sum() < 2 or sid == did: continue
            if sid in hp2i: s_type, s_idx = "home_player", hp2i[sid]
            elif sid in vp2i: s_type, s_idx = "visitor_player", vp2i[sid]
            else: continue
            if did in hp2i: d_type, d_idx = "home_player", hp2i[did]
            elif did in vp2i: d_type, d_idx = "visitor_player", vp2i[did]
            else: continue
            key = (s_type, "interact", d_type)
            if "edge_index" not in data[key]:
                data[key].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[key].edge_attr  = torch.empty((0, self.EDGE_DIM))
            data[key].edge_index = torch.cat(
                [data[key].edge_index,
                 torch.tensor([[s_idx], [d_idx]], dtype=torch.long)], 1)
            data[key].edge_attr = torch.cat([data[key].edge_attr, vec.unsqueeze(0)], 0)

        # team ↔ team edges
        total = torch.zeros(self.EDGE_DIM)
        for v in inter.values(): total += v
        for src, dst in (("home_team","visitor_team"), ("visitor_team","home_team")):
            et = (src, "versus", dst)
            data[et].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            data[et].edge_attr  = total.unsqueeze(0)

        # ensure every edge-type exists
        for et in self.ALL_EDGE_TYPES:
            if et not in data:
                data[et].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[et].edge_attr  = torch.empty((0, self.EDGE_DIM))

        data["home_team"].y = label
        return data

    def create_graphs(self):
        groups = list(self.df.groupby("GAME_ID"))
        if self.DEBUG_N: groups = groups[:self.DEBUG_N]
        graphs = [self.build_graph(g) for _, g in tqdm(groups)]
        print(f"Graphs {len(graphs)}  |  first_half_only = {self.FIRST_HALF_ONLY}")
        return graphs