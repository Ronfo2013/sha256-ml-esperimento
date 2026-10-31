"""Generazione dei dati: prefissi, nonce, hash (top64) ed etichette; split per prefisso.

Le feature usano SOLO ciò che è noto prima di calcolare l'hash: bit del nonce,
qualche statistica elementare del nonce, bit della coda del prefisso.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config
from hashing import success_from_top64, top64, top64_many

# ---------------------------------------------------------------- oracoli


def oracle_top64_many(name: str, prefixes, nonces_list, workers=None) -> list[np.ndarray]:
    """'sha256d' = doppio SHA-256 reale (hashlib).
    'piantato' = controllo positivo: SHA-256 reale, ma per i nonce con i 4 bit bassi a zero
    il valore è dimezzato, quindi la probabilità di successo raddoppia (a ogni k).
    È una funzione FINTA, costruita apposta perché un segnale ci sia."""
    tops = top64_many(prefixes, nonces_list, 64, True, workers)
    if name == "sha256d":
        return tops
    if name == "piantato":
        out = []
        for n, t in zip(nonces_list, tops):
            t = t.copy()
            mask = (np.asarray(n, dtype=np.uint64) & np.uint64(15)) == 0
            t[mask] >>= np.uint64(1)
            out.append(t)
        return out
    raise ValueError(f"oracolo sconosciuto: {name}")


def oracle_top64(name: str, prefix: bytes, nonces) -> np.ndarray:
    return oracle_top64_many(name, [prefix], [nonces], workers=1)[0]


def null_rate(name: str, k: int) -> float:
    """Tasso di successo atteso scegliendo a caso (ipotesi nulla)."""
    p = 2.0 ** -k
    if name == "sha256d":
        return p
    if name == "piantato":
        return p * (15 / 16) + 2 * p * (1 / 16)
    raise ValueError(name)


# ---------------------------------------------------------------- feature

N_NONCE_FEATURES = 64 + 4


def nonce_features(nonces: np.ndarray) -> np.ndarray:
    n = np.asarray(nonces, dtype=np.uint64)
    bits = ((n[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(np.float32)
    pop = bits.sum(axis=1, keepdims=True) / 64.0
    low = (n & np.uint64(0xFFFFFFFF)).astype(np.float64)[:, None] / 2**32
    high = (n >> np.uint64(32)).astype(np.float64)[:, None] / 2**32
    tz = np.zeros((len(n), 1), dtype=np.float32)
    nz = n != 0
    tz[nz, 0] = np.log2((n[nz] & (~n[nz] + np.uint64(1))).astype(np.float64)) / 64.0
    return np.hstack([bits, pop, low.astype(np.float32), high.astype(np.float32), tz])


def prefix_features(prefix: bytes, n_rows: int) -> np.ndarray:
    """Bit degli ultimi 12 byte del prefisso: stanno nello stesso blocco SHA del nonce."""
    tail = np.frombuffer(prefix[64:], dtype=np.uint8)
    bits = np.unpackbits(tail).astype(np.float32)
    return np.broadcast_to(bits, (n_rows, bits.size))


def features(prefix: bytes, nonces: np.ndarray, cfg: Config) -> np.ndarray:
    f = nonce_features(nonces)
    if cfg.use_prefix_features:
        f = np.hstack([f, prefix_features(prefix, len(f))])
    return f


# ---------------------------------------------------------------- split


@dataclass
class Split:
    prefixes: list[bytes]
    nonces: list[np.ndarray]
    top64: list[np.ndarray]


def _rng(cfg: Config, *tag: int) -> np.random.Generator:
    return np.random.default_rng([cfg.master_seed, *tag])


def make_prefixes(cfg: Config) -> dict[str, list[bytes]]:
    """Prefissi distinti per split, generati da stream di seed separati."""
    sizes = {"train": cfg.n_train_prefixes, "val": cfg.n_val_prefixes, "test": cfg.n_test_prefixes}
    out = {}
    for i, (name, n) in enumerate(sizes.items()):
        rng = _rng(cfg, 1, i)
        out[name] = [rng.bytes(cfg.prefix_len) for _ in range(n)]
    seen = set()
    for name, lst in out.items():
        s = set(lst)
        if seen & s or len(s) != len(lst):
            raise RuntimeError("prefisso ripetuto fra split: rigenerare con altro seed")
        seen |= s
    return out


def random_nonces(cfg: Config, split_id: int, idx: int, n: int) -> np.ndarray:
    return _rng(cfg, 2, split_id, idx).integers(0, 2**64 - 1, size=n, dtype=np.uint64,
                                                endpoint=True)


def build_split(cfg: Config, prefixes: list[bytes], split_id: int, n_nonces: int) -> Split:
    nonces = [random_nonces(cfg, split_id, i, n_nonces) for i in range(len(prefixes))]
    tops = oracle_top64_many(cfg.oracolo, prefixes, nonces, cfg.workers)
    return Split(prefixes, nonces, tops)


def to_xy(split: Split, cfg: Config, k_label: int, max_rows: int | None = None,
          seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matrice delle feature, etichette e indice del prefisso (per bootstrap a grappolo).
    Con max_rows sottocampiona solo i negativi: l'AUC (ordinamento) non cambia in attesa."""
    X = np.vstack([features(p, n, cfg) for p, n in zip(split.prefixes, split.nonces)])
    y = np.concatenate([success_from_top64(t, k_label) for t in split.top64]).astype(np.int8)
    g = np.concatenate([np.full(len(n), i, dtype=np.int32) for i, n in enumerate(split.nonces)])
    if max_rows is not None and len(y) > max_rows:
        rng = np.random.default_rng([cfg.master_seed, 3, seed])
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        keep_neg = rng.choice(neg, size=max(0, max_rows - len(pos)), replace=False)
        idx = np.sort(np.concatenate([pos, keep_neg]))
        X, y, g = X[idx], y[idx], g[idx]
    return X, y, g


def verify(cfg: Config, prefix: bytes, nonces: np.ndarray) -> np.ndarray:
    """Verifica indipendente con la funzione reale: ricalcola gli hash dei candidati scelti."""
    return oracle_top64(cfg.oracolo, prefix, nonces)


def sequential_top64(cfg: Config, prefixes: list[bytes], n: int) -> list[np.ndarray]:
    seq = [np.arange(n, dtype=np.uint64)] * len(prefixes)
    return oracle_top64_many(cfg.oracolo, prefixes, seq, cfg.workers)


__all__ = ["Split", "make_prefixes", "build_split", "to_xy", "features", "verify",
           "sequential_top64", "null_rate", "oracle_top64_many", "top64"]
