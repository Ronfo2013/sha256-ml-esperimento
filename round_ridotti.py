"""Controllo positivo strutturale: dove si perde la struttura di SHA-256 al crescere dei round.

Per ogni numero di round R e per SHA-256 singolo e doppio misuro l'AUC di un gradient boosting
che guarda i bit del nonce, in due regimi:
- stesso prefisso: addestramento e test su nonce diversi dello STESSO prefisso. Qui il leakage
  è voluto: misura se esiste struttura sfruttabile in assoluto (il caso più favorevole).
- prefissi nuovi: addestramento su alcuni prefissi, test su altri (il regime dell'esperimento).
Etichetta: hash < 2^(256-4) (successo con probabilità 1/16, per avere molti positivi).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from config import Config
from dataset import nonce_features
from hashing import success_from_top64, top64_many

K_STUDY = 4


def _auc(model, X, y) -> float:
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def study(cfg: Config) -> list[dict]:
    rng = np.random.default_rng([cfg.master_seed, 10])
    n_pre = cfg.round_prefixes
    prefixes = [rng.bytes(cfg.prefix_len) for _ in range(2 * n_pre)]
    nonces = [rng.integers(0, 2**64 - 1, size=cfg.round_nonces, dtype=np.uint64, endpoint=True)
              for _ in prefixes]
    feats = [nonce_features(n) for n in nonces]
    cut = int(cfg.round_nonces * 2 / 3)
    rows = []
    for double in (False, True):
        for R in cfg.round_list:
            tops = top64_many(prefixes, nonces, R, double, cfg.workers)
            ys = [success_from_top64(t, K_STUDY).astype(int) for t in tops]
            same = []
            for i in range(n_pre):
                X, y = feats[i], ys[i]
                if y[:cut].min() == y[:cut].max():
                    same.append(float("nan"))
                    continue
                m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
                m.fit(X[:cut], y[:cut])
                same.append(_auc(m, X[cut:], y[cut:]))
            Xtr, ytr = np.vstack(feats[:n_pre]), np.concatenate(ys[:n_pre])
            Xte, yte = np.vstack(feats[n_pre:]), np.concatenate(ys[n_pre:])
            cross = float("nan")
            if ytr.min() != ytr.max():
                m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xtr, ytr)
                cross = _auc(m, Xte, yte)
            rows.append(dict(funzione="doppio" if double else "singolo", round=R,
                             tasso=float(np.mean(np.concatenate(ys))), atteso=2.0 ** -K_STUDY,
                             auc_stesso_prefisso=float(np.nanmean(same)),
                             auc_stesso_prefisso_min=float(np.nanmin(same)),
                             auc_stesso_prefisso_max=float(np.nanmax(same)),
                             auc_prefissi_nuovi=cross))
            print(f"  round {R:2d} {'doppio ' if double else 'singolo'}: "
                  f"AUC stesso prefisso {rows[-1]['auc_stesso_prefisso']:.3f}, "
                  f"prefissi nuovi {cross:.3f}", flush=True)
    return rows
