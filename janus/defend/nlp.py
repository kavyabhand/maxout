"""NLP defend family, flags injected/manipulative catalog text before it
reaches an agent's context (the Mandate Firewall's content-scrub layer)
and, more generally, is the "phishing text / scam chat / injection
payload" detector in the 5-family ensemble.

Two tiers, both offline once trained:

- InjectionTextClassifier: TF-IDF + logistic regression. Trains in well
  under a second on the template corpus, supports incremental `.learn()`
  by refitting on an accumulated example buffer (cheap enough at this
  corpus scale that a real refit-per-round is simpler and more honest than
  a fake incremental-update mechanism). This is the always-available tier.
- DistilBertInjectionClassifier: an optional local fine-tune for a
  stronger result, same interface. Training is slower (CPU minutes, not
  seconds) so it's opt-in via train_distilbert(), not built by default.

Both score() to a float in [0, 1] and share the same scrub() contract the
firewall depends on, so the firewall can swap tiers without caring which
is active.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from janus.common.schemas import Verdict

QUARANTINE_PLACEHOLDER = "[content removed by Mandate Firewall: flagged as likely manipulation/injection]"


@dataclass
class ScrubResult:
    verdict: Verdict
    cleaned_text: str
    score: float
    reasons: list[str]


class InjectionTextClassifier:
    name = "tfidf_logreg"

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._texts: list[str] = []
        self._labels: list[int] = []

    def fit(self, texts: list[str], labels: list[int]) -> None:
        self._texts, self._labels = list(texts), list(labels)
        self._refit()

    def _refit(self) -> None:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipeline.fit(self._texts, self._labels)
        self._pipeline = pipeline

    def learn(self, texts: list[str], *, label: int = 1) -> None:
        """Live-learning: fold newly-discovered examples (e.g. a red-team
        payload that got through this round) into the training set and
        refit immediately, so the very next scrub call in the same closed-
        loop round can already catch a repeat or near-repeat."""

        self._texts.extend(texts)
        self._labels.extend([label] * len(texts))
        self._refit()

    def score(self, text: str) -> float:
        if self._pipeline is None:
            return 0.0
        proba = self._pipeline.predict_proba([text])[0]
        classes = list(self._pipeline.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else 0.0

    def scrub(self, text: str, *, block_threshold: float = 0.7, flag_threshold: float = 0.4) -> ScrubResult:
        s = self.score(text)
        if s >= block_threshold:
            return ScrubResult(Verdict.BLOCK, QUARANTINE_PLACEHOLDER, s, [f"classifier score {s:.3f} >= block threshold {block_threshold}"])
        if s >= flag_threshold:
            return ScrubResult(Verdict.FLAG, text, s, [f"classifier score {s:.3f} >= flag threshold {flag_threshold}"])
        return ScrubResult(Verdict.PASS, text, s, [])


class DistilBertInjectionClassifier:
    """Optional stronger tier. Loads a locally fine-tuned checkpoint (see
    train_distilbert() below); raises FileNotFoundError if none exists,
    which callers (firewall.py) treat as "fall back to the TF-IDF tier",
    matching the predecessor project's graceful-degradation pattern."""

    name = "distilbert"

    def __init__(self, checkpoint_dir: str) -> None:
        import os

        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"no DistilBERT checkpoint at {checkpoint_dir!r}")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self._model.eval()

    def score(self, text: str) -> float:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs.pop("token_type_ids", None)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        return float(probs[1])

    def scrub(self, text: str, *, block_threshold: float = 0.7, flag_threshold: float = 0.4) -> ScrubResult:
        s = self.score(text)
        if s >= block_threshold:
            return ScrubResult(Verdict.BLOCK, QUARANTINE_PLACEHOLDER, s, [f"classifier score {s:.3f} >= block threshold {block_threshold}"])
        if s >= flag_threshold:
            return ScrubResult(Verdict.FLAG, text, s, [f"classifier score {s:.3f} >= flag threshold {flag_threshold}"])
        return ScrubResult(Verdict.PASS, text, s, [])


def train_distilbert(texts: list[str], labels: list[int], out_dir: str, *, epochs: int = 4) -> dict:
    """Fine-tunes distilbert-base-uncased on the given corpus, entirely
    locally (downloads the base weights from the HF Hub once, then trains
    on CPU). Returns held-out eval metrics. Not called by default; the
    TF-IDF tier is the always-on baseline; this is an optional upgrade
    path exercised explicitly in the model-training phase."""

    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    x_train, x_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, padding=True, max_length=256)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items() if k != "token_type_ids"}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_ds, val_ds = TextDataset(x_train, y_train), TextDataset(x_val, y_val)

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=8,
        learning_rate=3e-5, eval_strategy="epoch", save_strategy="no", logging_strategy="no", report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    from sklearn.metrics import f1_score, precision_score, recall_score

    return {
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "n_val": len(y_val),
    }
