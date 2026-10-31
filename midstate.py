"""Punto 3: la struttura fra 4 e 5 round, con le feature che un miner calcola PRIMA del nonce.

Messaggio = prefisso (76 byte) + nonce (8 byte LE) = 84 byte, due blocchi SHA-256.
- Blocco 1 = prefisso[0:64]: non dipende dal nonce -> "midstate".
- Blocco 2 = prefisso[64:76] (W0..W2) + nonce (W3, W4) + padding. I round 0-2 del blocco 2 non
  toccano il nonce: lo stato dopo il round 2 ("s3") si calcola una volta per prefisso.
- Al round 3 entra W3. Tutto ciò che il round 3 somma a W3 è noto prima: lo chiamo D.
  A 4 round la prima parola dell'hash è esattamente (D_a + W3) mod 2^32 (verificato nei test).

Ipotesi pre-registrate (scritte il 31/10/2026, prima di guardare i risultati):
  H1  SHA singolo a 4 round con feature "precalcolo": AUC su prefissi nuovi nettamente > 0,5.
  H2  SHA singolo a 4 round con sole feature del nonce: AUC ~ 0,5 (manca D; vedi round_ridotti).
  H3  da 5 round in su, e per il doppio SHA a ogni R: AUC ~ 0,5 con qualunque feature.
Iperparametri fissi; test unilaterale AUC > 0,5 con errore standard da bootstrap a grappolo
sui prefissi; Bonferroni su tutte le celle (round x funzione x feature x modello).

Regola dell'esperimento principale rispettata: il modello non calcola mai l'hash. Le feature
"precalcolo" sono solo quantità indipendenti dal nonce; la somma con W3 la deve imparare.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import nonce_features
from hashing import _IV, _K, _compress, _rotr, sha256_numpy, success_from_top64

K_STUDY = 4
FEATURE_SETS = ("nonce", "midstate", "precalcolo")
MODELS = ("logistica", "gbm", "mlp")


def _words(b: np.ndarray) -> np.ndarray:
    """(n, 4m) uint8 -> (m, n) uint32 big-endian."""
    return np.ascontiguousarray(b).view(">u4").astype(np.uint32).T


def precompute(prefixes: list[bytes], rounds: int) -> dict[str, np.ndarray]:
    """Quantità indipendenti dal nonce, vettorizzate sui prefissi (con R round ovunque,
    come `hashing.top64_numpy`). Ritorna array (8, P) o (P,) uint32."""
    pre = np.frombuffer(b"".join(prefixes), dtype=np.uint8).reshape(len(prefixes), -1)
    with np.errstate(over="ignore"):
        iv = np.repeat(_IV[:, None], len(prefixes), axis=1)
        mid = _compress(iv, _words(pre[:, :64]), rounds)
        w = _words(pre[:, 64:76])                      # W0, W1, W2
        a, b, c, d, e, f, g, h = (mid[i].copy() for i in range(8))
        for t in range(3):
            S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            t1 = h + S1 + ((e & f) ^ (~e & g)) + _K[t] + w[t]
            t2 = (_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))
            h, g, f, e, d, c, b, a = g, f, e, d + t1, c, b, a, t1 + t2
        s3 = np.stack([a, b, c, d, e, f, g, h])
        # round 3 senza W3: t1 = C1 + W3, t2 = C2
        c1 = h + (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) + ((e & f) ^ (~e & g)) + _K[3]
        c2 = (_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))
        d_a = mid[0] + c1 + c2        # a 4 round: H0 = D_a + W3
        d_e = mid[4] + d + c1         # a 4 round: H4 = D_e + W3
    return dict(mid=mid, s3=s3, d=np.stack([d_a, d_e]))


def nonce_words(nonces: np.ndarray) -> np.ndarray:
    """(n,) uint64 -> (2, n) uint32: W3 e W4 come li vede SHA-256 (byte LE letti big-endian)."""
    nb = np.asarray(nonces, dtype=np.uint64).astype("<u8").view(np.uint8).reshape(-1, 8)
    return _words(nb)


def _bits(words: np.ndarray) -> np.ndarray:
    """(m, n) uint32 -> (n, 32m) float32."""
    sh = np.arange(32, dtype=np.uint32)
    return ((words.T[:, :, None] >> sh) & np.uint32(1)).reshape(words.shape[1], -1).astype(np.float32)


def _ints(words: np.ndarray) -> np.ndarray:
    return (words.T.astype(np.float64) / 2**32).astype(np.float32)


def build_features(fs: str, pc: dict, pidx: np.ndarray, nonces: np.ndarray) -> np.ndarray:
    """pidx: indice del prefisso per ogni riga; pc: output di `precompute`."""
    if fs not in FEATURE_SETS:
        raise ValueError(fs)
    cols = [nonce_features(nonces), _ints(nonce_words(nonces))]
    if fs in ("midstate", "precalcolo"):
        cols += [_ints(pc["mid"][:, pidx]), _ints(pc["s3"][:, pidx]), _bits(pc["s3"][:, pidx])]
    if fs == "precalcolo":
        cols += [_ints(pc["d"][:, pidx]), _bits(pc["d"][:, pidx])]
    return np.hstack(cols)


def labels(prefixes: list[bytes], pidx: np.ndarray, nonces: np.ndarray, rounds: int,
           double: bool, chunk: int = 50_000) -> np.ndarray:
    """top64 di tutte le righe, vettorizzato (stessa funzione di hashing.top64_numpy)."""
    out = np.empty(len(nonces), dtype=np.uint64)
    pre = np.frombuffer(b"".join(prefixes), dtype=np.uint8).reshape(len(prefixes), -1)
    for s in range(0, len(nonces), chunk):
        sl = slice(s, s + chunk)
        nb = nonces[sl].astype("<u8").view(np.uint8).reshape(-1, 8)
        dig = sha256_numpy(np.concatenate([pre[pidx[sl]], nb], axis=1), rounds)
        if double:
            dig = sha256_numpy(dig, rounds)
        out[sl] = dig[:, :8].copy().view(">u8").astype(np.uint64).ravel()
    return out


def make_model(name: str):
    if name == "logistica":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if name == "gbm":
        return HistGradientBoostingClassifier(max_iter=300, random_state=0)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200, random_state=0))
    raise ValueError(name)


def cluster_auc(y: np.ndarray, s: np.ndarray, pidx: np.ndarray, reps: int, seed) -> tuple:
    """AUC con IC ed errore standard da bootstrap a grappolo sui prefissi."""
    auc = float(roc_auc_score(y, s))
    groups = np.unique(pidx)
    order = np.argsort(pidx, kind="stable")
    starts = np.searchsorted(pidx[order], groups)
    ends = np.append(starts[1:], len(order))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([order[starts[g]:ends[g]] for g in pick])
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc_score(y[idx], s[idx]))
    boots = np.array(boots)
    return auc, float(boots.std(ddof=1)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def top_lift(y: np.ndarray, s: np.ndarray, pidx: np.ndarray, frac: float = 1 / 16) -> float:
    """Tasso di successo fra i primi `frac` candidati di ogni prefisso, diviso il tasso medio."""
    hits, n = 0, 0
    for p in np.unique(pidx):
        m = pidx == p
        k = max(1, int(round(m.sum() * frac)))
        top = np.argsort(-s[m], kind="stable")[:k]
        hits += int(y[m][top].sum())
        n += k
    return (hits / n) / y.mean() if y.mean() else float("nan")


def split(master_seed: int, n_train: int, n_test: int, per_prefix: int):
    rng = np.random.default_rng([master_seed, 30])
    prefixes = [rng.bytes(76) for _ in range(n_train + n_test)]
    if len(set(prefixes)) != len(prefixes):
        raise RuntimeError("prefissi duplicati")
    pidx = np.repeat(np.arange(len(prefixes)), per_prefix)
    nonces = rng.integers(0, 2**64 - 1, size=len(pidx), dtype=np.uint64, endpoint=True)
    is_test = pidx >= n_train
    return prefixes, pidx, nonces, is_test


def study(master_seed: int = 20261031, n_train: int = 3000, n_test: int = 1000,
          per_prefix: int = 64, round_list=(4, 5, 6, 8), double_rounds=(4, 5),
          models=MODELS, feature_sets=FEATURE_SETS, reps: int = 500) -> list[dict]:
    prefixes, pidx, nonces, is_test = split(master_seed, n_train, n_test, per_prefix)
    cells = [(False, r) for r in round_list] + [(True, r) for r in double_rounds]
    rows = []
    for double, R in cells:
        y = success_from_top64(labels(prefixes, pidx, nonces, R, double), K_STUDY).astype(int)
        pc = precompute(prefixes, R)
        for fs in feature_sets:
            X = build_features(fs, pc, pidx, nonces)
            for mname in models:
                t = time.perf_counter()
                m = make_model(mname).fit(X[~is_test], y[~is_test])
                s = m.predict_proba(X[is_test])[:, 1]
                auc, se, lo, hi = cluster_auc(y[is_test], s, pidx[is_test], reps,
                                              [master_seed, 31, R, int(double)])
                rows.append(dict(funzione="doppio" if double else "singolo", round=R,
                                 feature=fs, modello=mname, auc=auc, auc_se=se, ic_basso=lo,
                                 ic_alto=hi, lift_top16=top_lift(y[is_test], s, pidx[is_test]),
                                 tasso=float(y.mean()), n_feature=X.shape[1],
                                 secondi=time.perf_counter() - t))
                r = rows[-1]
                print(f"  {r['funzione']:7s} R={R} {fs:10s} {mname:9s} AUC {auc:.3f} "
                      f"[{lo:.3f}-{hi:.3f}] lift {r['lift_top16']:.2f} ({r['secondi']:.0f} s)",
                      flush=True)
    return rows
