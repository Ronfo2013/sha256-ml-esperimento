import hashlib
import os

import numpy as np

from baselines import first_hit, random_search, sequential
from config import Config, rapido
from dataset import (build_split, features, make_prefixes, null_rate, oracle_top64_many,
                     to_xy, verify)
from evaluate import auc_hanley, bootstrap_ratio_ci, binom_greater, bootstrap_rate_ci, min_detectable_lift
from hashing import success_from_top64, top64_hashlib, top64_numpy
from model import Scorer

CFG = rapido(Config()).with_(n_train_prefixes=20, n_val_prefixes=5, n_test_prefixes=5,
                             nonces_per_train_prefix=512, workers=1)


def test_sha_singolo_numpy_coincide_con_hashlib():
    prefix = os.urandom(76)
    nonces = np.arange(300, dtype=np.uint64)
    ref = np.array([int.from_bytes(hashlib.sha256(prefix + int(n).to_bytes(8, "little"))
                                   .digest()[:8], "big") for n in nonces], dtype=np.uint64)
    assert np.array_equal(top64_hashlib(prefix, nonces, double=False), ref)
    assert np.array_equal(top64_numpy(prefix, nonces, 64, double=False), ref)


def test_split_per_prefisso_senza_sovrapposizioni():
    pre = make_prefixes(CFG)
    a, b, c = map(set, (pre["train"], pre["val"], pre["test"]))
    assert not (a & b or a & c or b & c)


def test_dataset_riproducibile_dal_seed():
    pre1, pre2 = make_prefixes(CFG), make_prefixes(CFG)
    assert pre1 == pre2
    s1 = build_split(CFG, pre1["train"][:3], 0, 100)
    s2 = build_split(CFG, pre2["train"][:3], 0, 100)
    assert all(np.array_equal(x, y) for x, y in zip(s1.top64, s2.top64))


def test_feature_non_dipendono_dall_hash():
    # stesse feature per prefissi che differiscono solo nella parte non usata (primi 64 byte)
    p1 = bytes(64) + bytes(range(12))
    p2 = bytes([1] * 64) + bytes(range(12))
    n = np.arange(50, dtype=np.uint64)
    assert np.array_equal(features(p1, n, CFG), features(p2, n, CFG))
    assert not np.array_equal(top64_hashlib(p1, n), top64_hashlib(p2, n))


def test_verify_usa_la_funzione_reale():
    prefix = os.urandom(76)
    n = np.arange(64, dtype=np.uint64)
    assert np.array_equal(verify(CFG, prefix, n), top64_hashlib(prefix, n))


def test_oracolo_piantato_raddoppia_il_tasso():
    rng = np.random.default_rng(0)
    pre = [rng.bytes(76) for _ in range(8)]
    nonces = [np.arange(40000, dtype=np.uint64)] * 8
    tops = np.concatenate(oracle_top64_many("piantato", pre, nonces, workers=1))
    allnon = np.concatenate(nonces)
    y = success_from_top64(tops, 4)
    mask = (allnon & np.uint64(15)) == 0
    assert abs(y[mask].mean() - 2 / 16) < 0.01
    assert abs(y[~mask].mean() - 1 / 16) < 0.005
    assert abs(null_rate("piantato", 4) - y.mean()) < 0.003


def test_to_xy_sottocampiona_solo_negativi():
    s = build_split(CFG, make_prefixes(CFG)["train"][:10], 0, 512)
    X, y, _ = to_xy(s, CFG, 4)
    Xs, ys, _ = to_xy(s, CFG, 4, max_rows=1000)
    assert ys.sum() == y.sum() and len(ys) == 1000


def test_baseline():
    s = np.array([0, 0, 1, 0, 1], dtype=bool)
    assert first_hit(s) == 3 and first_hit(np.zeros(3, bool)) is None
    assert sequential(s, 3) == (1, 3)
    k, att, chosen = random_search(s, 5, [1])
    assert k == 2 and len(chosen) == 5


def test_classe_maggioritaria_ordina_a_caso():
    X = np.zeros((1000, 3))
    sc = Scorer("maggioritaria", 0).fit(X, np.r_[np.zeros(990), np.ones(10)].astype(int))
    o1, o2 = sc.rank(X, [1]), sc.rank(X, [2])
    assert sorted(o1) == list(range(1000)) and not np.array_equal(o1, o2)


def test_statistica():
    assert binom_greater(100, 1000, 0.1) > 0.4
    assert binom_greater(200, 1000, 0.1) < 1e-10
    lo, hi = bootstrap_rate_ci(np.full(100, 5), 50, 500, [0])
    assert lo == hi == 0.1
    mde = min_detectable_lift(4_000_000, 2 ** -12, 0.05)
    assert 0.05 < mde < 0.3
    a, se = auc_hanley(np.r_[np.zeros(500), np.ones(500)], np.r_[np.zeros(500), np.ones(500)])
    assert a == 1.0


def test_lift_senza_successi_nel_riferimento():
    lo, hi = bootstrap_ratio_ci(np.array([0, 1, 0]), np.zeros(3, dtype=int), 200, [0])
    assert np.isnan(lo) and np.isnan(hi)
