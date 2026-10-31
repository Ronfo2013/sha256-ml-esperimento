import numpy as np
import pytest

from hashing import sha256_numpy, top64_hashlib, top64_numpy
from midstate import (FEATURE_SETS, build_features, labels, nonce_words, precompute, split)


@pytest.fixture(scope="module")
def dati():
    prefixes, pidx, nonces, is_test = split(7, 6, 4, 16)
    return prefixes, pidx, nonces, is_test


def _per_prefisso(prefixes, pidx, nonces, rounds, double):
    out = np.empty(len(nonces), dtype=np.uint64)
    for p in np.unique(pidx):
        m = pidx == p
        out[m] = top64_numpy(prefixes[p], nonces[m], rounds, double)
    return out


def test_somma_a_4_round(dati):
    prefixes, pidx, nonces, _ = dati
    pc = precompute(prefixes, 4)
    w3 = nonce_words(nonces)[0]
    h0 = (pc["d"][0][pidx] + w3).astype(np.uint32)
    h4 = (pc["d"][1][pidx] + w3).astype(np.uint32)
    top = labels(prefixes, pidx, nonces, 4, False)
    assert np.array_equal(h0, (top >> np.uint64(32)).astype(np.uint32))
    # quinta parola dell'hash (H4) con sha256_numpy, riga per riga
    for i in range(len(nonces)):
        msg = np.frombuffer(prefixes[pidx[i]] + int(nonces[i]).to_bytes(8, "little"), np.uint8)[None]
        dig = sha256_numpy(msg, 4)[0]
        assert int.from_bytes(dig[16:20].tobytes(), "big") == int(h4[i])


@pytest.mark.parametrize("rounds", [4, 5])
@pytest.mark.parametrize("double", [False, True])
def test_labels_coincide_con_top64_numpy(dati, rounds, double):
    prefixes, pidx, nonces, _ = dati
    assert np.array_equal(labels(prefixes, pidx, nonces, rounds, double, chunk=20),
                          _per_prefisso(prefixes, pidx, nonces, rounds, double))


def test_labels_64_round_coincide_con_hashlib():
    prefixes, pidx, nonces, _ = split(3, 3, 2, 8)
    atteso = np.concatenate([top64_hashlib(prefixes[p], nonces[pidx == p], True)
                             for p in np.unique(pidx)])
    assert np.array_equal(labels(prefixes, pidx, nonces, 64, True), atteso)


def test_feature_di_prefisso_costanti_e_indipendenti_dall_hash(dati):
    prefixes, pidx, nonces, _ = dati
    pc = precompute(prefixes, 4)
    X = build_features("precalcolo", pc, pidx, nonces)
    Xn = build_features("nonce", pc, pidx, nonces)
    n_nonce = Xn.shape[1]
    assert X.shape[1] > n_nonce
    assert np.array_equal(X[:, :n_nonce], Xn)
    for p in np.unique(pidx):
        blocco = X[pidx == p, n_nonce:]
        assert (blocco == blocco[0]).all()
    # non dipendono dal nonce: stesse colonne con nonce diversi
    X2 = build_features("precalcolo", pc, pidx, nonces[::-1].copy())
    assert np.array_equal(X[:, n_nonce:], X2[:, n_nonce:])
    # e dipendono dal solo prefisso, non dall'hash: sono calcolate senza il nonce
    assert np.array_equal(precompute(prefixes, 4)["d"], pc["d"])
    with pytest.raises(ValueError):
        build_features("inesistente", pc, pidx, nonces)
    assert set(FEATURE_SETS) == {"nonce", "midstate", "precalcolo"}


def test_split_senza_prefissi_in_comune():
    prefixes, pidx, nonces, is_test = split(1, 20, 8, 5)
    assert not set(pidx[~is_test]) & set(pidx[is_test])
    assert len(prefixes) == 28 and is_test.sum() == 8 * 5
    assert len(set(prefixes)) == len(prefixes)
