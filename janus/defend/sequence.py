"""Sequence defend family; a small transformer encoder over per-entity
event sequences (janus/generate/sequence/simulator.py), catching
behavioral-drift patterns a single-row model structurally can't see:
card-testing bursts and structuring cadence. Each event step is
(delta_time, log_amount, event_type); a learned positional/temporal
encoding plus a few self-attention layers pool to a per-entity risk score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from janus.common.metrics import EvalReport, evaluate
from janus.generate.sequence.simulator import EVENT_TO_IDX, EntitySequence

MAX_LEN = 64


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[EntitySequence]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        n = min(len(seq.steps), MAX_LEN)
        delta_t = np.zeros(MAX_LEN, dtype=np.float32)
        log_amt = np.zeros(MAX_LEN, dtype=np.float32)
        event_idx = np.zeros(MAX_LEN, dtype=np.int64)
        mask = np.zeros(MAX_LEN, dtype=np.bool_)

        prev_step = seq.steps[0] if seq.steps else 0
        for i in range(n):
            delta_t[i] = seq.steps[i] - prev_step
            prev_step = seq.steps[i]
            log_amt[i] = np.log1p(max(seq.amounts[i], 0.0))
            event_idx[i] = EVENT_TO_IDX.get(seq.event_types[i], 0)
            mask[i] = True

        return {
            "delta_t": torch.tensor(delta_t),
            "log_amt": torch.tensor(log_amt),
            "event_idx": torch.tensor(event_idx),
            "mask": torch.tensor(mask),
            "label": torch.tensor(float(seq.label)),
        }


class SequenceTransformer(nn.Module):
    def __init__(self, n_event_types: int = 4, d_model: int = 32, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.event_embed = nn.Embedding(n_event_types, d_model)
        self.numeric_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Embedding(MAX_LEN, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, delta_t, log_amt, event_idx, mask):
        numeric = torch.stack([delta_t, log_amt], dim=-1)
        x = self.event_embed(event_idx) + self.numeric_proj(numeric)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pos_embed(positions)

        pad_mask = ~mask  # TransformerEncoder expects True = ignore
        encoded = self.encoder(x, src_key_padding_mask=pad_mask)

        mask_f = mask.unsqueeze(-1).float()
        pooled = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled).squeeze(-1)


@dataclass
class SequenceModelResult:
    model: SequenceTransformer
    eval_report: EvalReport


def train_sequence_transformer(train_seqs: list[EntitySequence], test_seqs: list[EntitySequence], *, epochs: int = 15, lr: float = 1e-3, random_state: int = 42) -> SequenceModelResult:
    torch.manual_seed(random_state)
    model = SequenceTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    labels = np.array([s.label for s in train_seqs])
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(SequenceDataset(train_seqs), batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch["delta_t"], batch["log_amt"], batch["event_idx"], batch["mask"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()

    model.eval()
    test_loader = DataLoader(SequenceDataset(test_seqs), batch_size=64, shuffle=False)
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["delta_t"], batch["log_amt"], batch["event_idx"], batch["mask"])
            all_scores.append(torch.sigmoid(logits).numpy())
            all_labels.append(batch["label"].numpy())

    scores = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    report = evaluate(y_true, scores)
    return SequenceModelResult(model=model, eval_report=report)
