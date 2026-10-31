"""Esegue l'intero esperimento e scrive il report.

    .venv/bin/python run_experiment.py            # completo (k=12 e 16, controlli)
    .venv/bin/python run_experiment.py --rapido   # prova veloce
    .venv/bin/python run_experiment.py --k 12 16 20
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

import report
import round_ridotti
from config import Config, rapido
from dataset import build_split, make_prefixes, null_rate, oracle_top64_many, random_nonces, \
    sequential_top64, to_xy
from evaluate import TestBench, compare, hash_seconds_per_candidate, min_detectable_lift
from model import Scorer


def learning_curve(cfg, Xtr, ytr, Xva, yva) -> list[dict]:
    out = []
    rng = np.random.default_rng([cfg.master_seed, 11])
    perm = rng.permutation(len(ytr))
    for frac in cfg.learning_curve_fracs:
        idx = perm[: max(1000, int(frac * len(ytr)))]
        row = dict(frazione=frac, righe=len(idx))
        for shuffle in (False, True):
            sc = Scorer("gbm", 0).fit(Xtr[idx], ytr[idx], shuffle_labels=shuffle)
            row["auc_val_mescolato" if shuffle else "auc_val"] = float(
                roc_auc_score(yva, sc.score(Xva)))
        out.append(row)
    return out


def run_one(cfg: Config) -> dict:
    t0 = time.perf_counter()
    print(f"\n=== oracolo={cfg.oracolo} k={cfg.k} (k_train={cfg.k_train})", flush=True)
    pre = make_prefixes(cfg)
    train = build_split(cfg, pre["train"], 0, cfg.nonces_per_train_prefix)
    val = build_split(cfg, pre["val"], 1, cfg.nonces_per_train_prefix)
    Xtr, ytr, _ = to_xy(train, cfg, cfg.k_train, cfg.max_train_rows, seed=0)
    Xva, yva, _ = to_xy(val, cfg, cfg.k_train, cfg.max_train_rows, seed=1)
    print(f"  train {len(ytr)} righe ({ytr.sum()} positivi a k_train), val {len(yva)}", flush=True)

    pools = [random_nonces(cfg, 2, i, cfg.n_candidates) for i in range(len(pre["test"]))]
    pool_tops = oracle_top64_many(cfg.oracolo, pre["test"], pools, cfg.workers)
    seq_tops = sequential_top64(cfg, pre["test"], cfg.n_candidates)
    bench = TestBench(cfg, pre["test"], pools, pool_tops, seq_tops)
    print(f"  test: {len(pools)} prefissi nuovi x {cfg.n_candidates} candidati, "
          f"verifico i primi {cfg.top_m}", flush=True)

    seq = bench.baseline_sequential()
    randoms = {s: bench.baseline_random(s) for s in cfg.model_seeds}
    methods, val_auc = [], {}
    for name in list(cfg.models) + ["gbm_mescolato"]:
        for s in cfg.model_seeds:
            sc = Scorer(name, s).fit(Xtr, ytr, shuffle_labels=(name == "gbm_mescolato"))
            val_auc[f"{name}/{s}"] = float(roc_auc_score(yva, sc.score(Xva))) \
                if name != "maggioritaria" else 0.5
            res = bench.run_model(sc, f"{name} (seed {s})")
            res.extra.update(modello=name, seed=s, auc_val=val_auc[f"{name}/{s}"])
            methods.append((res, s))
            print(f"  {res.name:26s} successi {res.total_success:5d}/{res.total_verified} "
                  f"AUC val {val_auc[f'{name}/{s}']:.3f}", flush=True)

    n_tests = len(methods)
    rows = [compare(cfg, m, randoms[s], n_tests, i) for i, (m, s) in enumerate(methods)]
    base_rows = [compare(cfg, seq, randoms[cfg.model_seeds[0]], n_tests, 900)] + \
                [compare(cfg, r, randoms[s], n_tests, 901 + s) for s, r in randoms.items()]
    for r in base_rows:
        r["significativo"] = None  # le baseline non sono ipotesi da testare
    n_ver = cfg.top_m * cfg.n_test_prefixes
    return dict(
        oracolo=cfg.oracolo, k=cfg.k, k_train=cfg.k_train,
        tasso_nullo=null_rate(cfg.oracolo, cfg.k),
        mde=min_detectable_lift(n_ver, null_rate(cfg.oracolo, cfg.k), cfg.alpha / n_tests),
        verificati_per_metodo=n_ver, confronti=n_tests,
        righe_train=int(len(ytr)), positivi_train=int(ytr.sum()),
        modelli=rows, baseline=base_rows,
        tentativi={"sequenziale": seq.attempts,
                   "casuale": randoms[cfg.model_seeds[0]].attempts,
                   **{m.name: m.attempts for m, _ in methods
                      if m.extra["modello"] in ("gbm", "logistica") and m.extra["seed"] == 0}},
        curva=learning_curve(cfg, Xtr, ytr, Xva, yva),
        hash_s=hash_seconds_per_candidate(pre["test"][0]),
        secondi=time.perf_counter() - t0,
        controlli=dict(prefissi_condivisi=len(set(pre["train"]) & set(pre["test"]))
                       + len(set(pre["val"]) & set(pre["test"]))
                       + len(set(pre["train"]) & set(pre["val"])),
                       verifica_incoerenze=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[12, 16])
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--seed", type=int, default=Config.master_seed)
    ap.add_argument("--out", default="risultati")
    a = ap.parse_args()
    cfg = Config(master_seed=a.seed)
    if a.rapido:
        cfg = rapido(cfg)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    runs = [run_one(cfg.with_(oracolo="piantato", k=a.k[0]))]
    runs += [run_one(cfg.with_(oracolo="sha256d", k=k)) for k in a.k]
    print("\n=== studio dei round ridotti", flush=True)
    rounds = round_ridotti.study(cfg)
    meta = dict(comando="run_experiment.py " + " ".join(
                    (["--rapido"] if a.rapido else []) + ["--k", *map(str, a.k), "--seed",
                                                          str(a.seed)]),
                config=cfg.as_dict(), python=platform.python_version(),
                numpy=np.__version__, sklearn=sklearn.__version__,
                macchina=f"{platform.machine()} {os.cpu_count()} core",
                secondi_totali=time.perf_counter() - t0)
    data = dict(meta=meta, corse=runs, round=rounds)
    (out / "risultati.json").write_text(json.dumps(data, indent=1, default=str))
    report.write(data, out)
    print(f"\nReport: {out / 'report.md'}  ({meta['secondi_totali']:.0f} s)")


if __name__ == "__main__":
    main()
