"""GNN defend family: GraphSAGE over an IEEE-CIS transaction/entity graph,
concatenated with raw tabular features into XGBoost as the hybrid decision
(the architecture a prior GNN-only vs. hybrid ablation validated: hybrid
PR-AUC 0.589 vs. GNN-alone 0.362 on the full 590k-row graph; the
relational embeddings and the tabular splitting are complementary, not
redundant).

IEEE-CIS has no explicit account/merchant/device ID field, it's
anonymized. The closest real proxies are `card1` (the strongest
card-identity proxy per the original competition's own winning
solutions), `addr1` (billing region), `P_emaildomain`, and `DeviceInfo`.
Two transactions sharing a `card1` value are, functionally, "the same
account" for graph purposes.

Graph shape: one homogeneous PyG graph, transaction nodes (real tabular
features + label) and entity nodes (one per unique value per linkage
field; features = a one-hot "which entity type" indicator, since an
entity node has no transaction-level features of its own). Edges connect
a transaction to every entity node it shares a value with, undirected. A
GraphSAGE encoder run over this graph lets fraud signal propagate between
transactions sharing a card/device/email even when their own raw features
look unremarkable individually, multi-hop mule-ring-style topology a
row-independent GBM structurally cannot see.

This runs entirely locally (CPU); the predecessor project ran this exact
architecture on Kaggle GPU notebooks; here it's one local module with no
notebook/GPU dependency, at the cost of slower training. A CUDA-usability
probe still exists so this transparently uses a GPU if one is actually
available and compatible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from janus.common.metrics import EvalReport, evaluate

ENTITY_COLS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]
NUMERIC_FEATURE_PREFIXES = ("TransactionAmt", "C", "D", "V")
MAX_NUMERIC_FEATURES = 64
EMBEDDING_DIM = 32
HIDDEN_DIM = 128


@dataclass
class GraphBuildResult:
    data: Data
    n_transactions: int
    feature_names: list[str]


def select_numeric_columns(df: pd.DataFrame, max_features: int = MAX_NUMERIC_FEATURES) -> list[str]:
    candidates = [c for c in df.columns if c.startswith(NUMERIC_FEATURE_PREFIXES) and pd.api.types.is_numeric_dtype(df[c])]
    # Prefer the fewest missing values, IEEE-CIS's V-columns are notoriously
    # sparse; a naive first-N pick would grab mostly-NaN features.
    non_null_counts = df[candidates].notna().sum().sort_values(ascending=False)
    return list(non_null_counts.head(max_features).index)


def build_transaction_entity_graph(df: pd.DataFrame, numeric_cols: list[str], label_col: str = "isFraud") -> GraphBuildResult:
    df = df.reset_index(drop=True)
    n_txn = len(df)

    txn_features = df[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
    mu, sigma = txn_features.mean(axis=0), txn_features.std(axis=0) + 1e-6
    txn_features = (txn_features - mu) / sigma

    entity_dim = len(ENTITY_COLS)
    txn_block = np.concatenate([txn_features, np.zeros((n_txn, entity_dim), dtype=np.float32)], axis=1)

    node_features = [txn_block]
    edges_src, edges_dst = [], []
    next_node_id = n_txn

    for entity_idx, col in enumerate(ENTITY_COLS):
        if col not in df.columns:
            continue
        value_to_node: dict = {}
        entity_rows = []
        for row_idx, val in enumerate(df[col].to_numpy()):
            if pd.isna(val):
                continue
            key = (col, val)
            if key not in value_to_node:
                value_to_node[key] = next_node_id
                next_node_id += 1
                onehot = np.zeros(entity_dim, dtype=np.float32)
                onehot[entity_idx] = 1.0
                entity_rows.append(np.concatenate([np.zeros(len(numeric_cols), dtype=np.float32), onehot]))
            node_id = value_to_node[key]
            edges_src.append(row_idx)
            edges_dst.append(node_id)
        if entity_rows:
            node_features.append(np.vstack(entity_rows))

    x = torch.tensor(np.concatenate(node_features, axis=0), dtype=torch.float)
    src = torch.tensor(edges_src + edges_dst, dtype=torch.long)
    dst = torch.tensor(edges_dst + edges_src, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    y = torch.full((x.size(0),), -1, dtype=torch.long)
    if label_col in df.columns:
        y[:n_txn] = torch.tensor(df[label_col].fillna(0).astype(int).to_numpy(), dtype=torch.long)

    return GraphBuildResult(data=Data(x=x, edge_index=edge_index, y=y), n_transactions=n_txn, feature_names=numeric_cols)


class JanusSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = HIDDEN_DIM, embedding_dim: int = EMBEDDING_DIM, num_classes: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)

    def embed(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

    def forward(self, x, edge_index):
        emb = self.embed(x, edge_index)
        return self.classifier(F.relu(emb)), emb


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            probe = torch.zeros(1, device="cuda")
            _ = probe + 1
            return torch.device("cuda")
        except RuntimeError:
            pass
    if torch.backends.mps.is_available():
        try:
            probe = torch.zeros(1, device="mps")
            _ = probe + 1
            return torch.device("mps")
        except RuntimeError:
            pass
    return torch.device("cpu")


@dataclass
class GnnHybridResult:
    gnn_eval: EvalReport
    hybrid_eval: EvalReport
    model_state_dict: dict
    xgb_model: xgb.XGBClassifier
    device: str
    n_nodes: int
    n_edges: int
    epochs: int


def train_gnn_hybrid(df: pd.DataFrame, *, epochs: int = 60, lr: float = 0.005, random_state: int = 42) -> GnnHybridResult:
    numeric_cols = select_numeric_columns(df)
    graph = build_transaction_entity_graph(df, numeric_cols)
    n_txn = graph.n_transactions

    device = _select_device()
    data = graph.data.to(device)

    perm = np.random.RandomState(random_state).permutation(n_txn)
    n_test = int(n_txn * 0.2)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True

    model = JanusSAGE(in_channels=data.x.shape[1]).to(device)
    class_counts = torch.bincount(data.y[train_mask])
    class_weights = (class_counts.sum() / (2.0 * class_counts.float())).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(data.x, data.edge_index)
        gnn_scores = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        embeddings_np = embeddings.cpu().numpy()

    y_true_test = data.y[test_idx].cpu().numpy()
    gnn_eval = evaluate(y_true_test, gnn_scores[test_idx])

    hybrid_features = np.concatenate([df[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32), embeddings_np[:n_txn]], axis=1)
    x_train, x_test = hybrid_features[train_idx], hybrid_features[test_idx]
    y_train, y_test = df["isFraud"].to_numpy()[train_idx], df["isFraud"].to_numpy()[test_idx]

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", tree_method="hist", random_state=random_state, n_jobs=-1,
    )
    xgb_model.fit(x_train, y_train)
    xgb_scores = xgb_model.predict_proba(x_test)[:, 1]
    hybrid_eval = evaluate(y_test, xgb_scores)

    return GnnHybridResult(
        gnn_eval=gnn_eval, hybrid_eval=hybrid_eval, model_state_dict=model.state_dict(),
        xgb_model=xgb_model, device=str(device), n_nodes=data.num_nodes, n_edges=data.num_edges, epochs=epochs,
    )
