"""Baseline di ricerca: sequenziale, casuale con seed. (La classe maggioritaria sta in model.py.)

Tutte lavorano su un prefisso alla volta e restituiscono:
- successi fra i primi M hash verificati
- posizione del primo successo (tentativi), o None se non trovato entro il limite
"""
from __future__ import annotations

import numpy as np


def first_hit(success_in_order: np.ndarray) -> int | None:
    idx = np.flatnonzero(success_in_order)
    return int(idx[0]) + 1 if idx.size else None


def sequential(success_seq: np.ndarray, m: int) -> tuple[int, int | None]:
    """success_seq: esito dei nonce 0, 1, 2, ... nell'ordine."""
    return int(success_seq[:m].sum()), first_hit(success_seq)


def random_order(n: int, seed_seq) -> np.ndarray:
    return np.random.default_rng(seed_seq).permutation(n)


def random_search(success_pool: np.ndarray, m: int, seed_seq) -> tuple[int, int | None, np.ndarray]:
    """Ordine casuale dei candidati del pool (stesso pool che vede il modello)."""
    order = random_order(len(success_pool), seed_seq)
    s = success_pool[order]
    return int(s[:m].sum()), first_hit(s), order[:m]
