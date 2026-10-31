"""Modelli: classe maggioritaria, regressione logistica, gradient boosting, piccola rete.

Iperparametri FISSATI a priori (nessuna ricerca sul test): cambiarli dopo aver visto
i risultati sarebbe una forma di p-hacking.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, seed: int):
    if name == "maggioritaria":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistica":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1))
    if name in ("gbm", "gbm_mescolato"):
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                              early_stopping=False, random_state=seed)
    if name == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                           batch_size=1024, max_iter=30, early_stopping=True,
                                           n_iter_no_change=4, random_state=seed))
    raise ValueError(name)


class Scorer:
    """Modello addestrato + ordinamento dei candidati. Il modello non calcola mai hash."""

    def __init__(self, name: str, seed: int):
        self.name, self.seed = name, seed
        self.model = make_model(name, seed)
        self.train_seconds = 0.0
        self.inference_seconds = 0.0
        self.inference_rows = 0

    def fit(self, X: np.ndarray, y: np.ndarray, shuffle_labels: bool = False) -> "Scorer":
        if shuffle_labels:  # controllo di sanità: etichette mescolate -> deve tornare al caso
            y = np.random.default_rng([self.seed, 99]).permutation(y)
        t = time.perf_counter()
        self.model.fit(X, y)
        self.train_seconds = time.perf_counter() - t
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        proba = self.model.predict_proba(X)
        self.inference_seconds += time.perf_counter() - t
        self.inference_rows += len(X)
        return proba[:, 1] if proba.shape[1] == 2 else np.zeros(len(X))

    def rank(self, X: np.ndarray, tie_seed) -> np.ndarray:
        """Indici dal più promettente al meno. Pareggi rotti a caso (con seed): un modello
        costante, come la classe maggioritaria, equivale così alla ricerca casuale."""
        s = self.score(X)
        jitter = np.random.default_rng(tie_seed).random(len(s))
        return np.lexsort((jitter, -s))

    @property
    def seconds_per_candidate(self) -> float:
        return self.inference_seconds / max(1, self.inference_rows)
