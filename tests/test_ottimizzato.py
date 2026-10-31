import hashlib
import os

import numpy as np
import pytest

from ottimizzato import (asicboost_cost, naive_cost, ops, optimized_cost, prepare, scan,
                         scan_asicboost)


def _ref_digest(h76: bytes, n: int) -> bytes:
    hdr = h76 + int(n).to_bytes(4, "little")
    return hashlib.sha256(hashlib.sha256(hdr).digest()).digest()


def _ref_success(h76: bytes, n: int, k: int) -> bool:
    """Definizione letterale Bitcoin: hash come intero little-endian < 2^(256-k)."""
    return int.from_bytes(_ref_digest(h76, n), "little") < (1 << (256 - k))


def test_header_genesi_bitcoin():
    hdr = bytes.fromhex(
        "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd"
        "7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c")
    d = hashlib.sha256(hashlib.sha256(hdr).digest()).digest()
    assert d[::-1].hex() == "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
    nonce = int.from_bytes(hdr[76:], "little")
    assert scan(prepare(hdr[:76]), np.array([nonce, nonce + 1], dtype=np.uint32), 32).tolist() \
        == [True, _ref_success(hdr[:76], nonce + 1, 32)]


@pytest.mark.parametrize("k", [1, 4, 8])
def test_scan_uguale_a_hashlib(k):
    h76 = os.urandom(76)
    nonces = np.random.default_rng(k).integers(0, 2**32, 3000, dtype=np.uint64).astype(np.uint32)
    got = scan(prepare(h76), nonces, k).tolist()
    assert got == [_ref_success(h76, int(n), k) for n in nonces]


def test_scan_k_alto_su_tanti_nonce():
    h76 = os.urandom(76)
    nonces = np.arange(200_000, dtype=np.uint32)
    got = scan(prepare(h76), nonces, 12)
    ref = np.array([_ref_success(h76, int(n), 12) for n in nonces])
    assert np.array_equal(got, ref) and ref.sum() > 20


def test_asicboost_uguale_a_hashlib():
    tail = os.urandom(12)
    heads = [os.urandom(64) + tail for _ in range(4)]   # stesso blocco 2, midstate diversi
    nonces = np.arange(2000, dtype=np.uint32)
    res = scan_asicboost([prepare(h) for h in heads], nonces, 4)
    for h, r in zip(heads, res):
        assert r.tolist() == [_ref_success(h, int(n), 4) for n in nonces]


def test_asicboost_rifiuta_blocchi_2_diversi():
    with pytest.raises(ValueError):
        scan_asicboost([prepare(os.urandom(76)), prepare(os.urandom(76))],
                       np.arange(3, dtype=np.uint32), 4)


def test_costi_coerenti():
    assert ops(naive_cost()) > ops(optimized_cost()) > ops(asicboost_cost(4))
    assert asicboost_cost(1) == optimized_cost()
