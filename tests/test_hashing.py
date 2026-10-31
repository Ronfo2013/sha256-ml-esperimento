import hashlib
import os

import numpy as np
import pytest

from hashing import (double_sha256, header_hash, is_success, sha256_numpy, success_from_top64,
                     top64, top64_hashlib, top64_numpy)

# Vettori di test ufficiali NIST FIPS 180-2
NIST = {
    b"": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    b"abc": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq":
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
}

# Header del blocco genesi di Bitcoin: vettore noto di doppio SHA-256
GENESIS = bytes.fromhex(
    "01000000" + "00" * 32
    + "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
    + "29ab5f49" + "ffff001d" + "1dac2b7c")
GENESIS_HASH = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"


@pytest.mark.parametrize("msg,expected", NIST.items())
def test_hashlib_nist(msg, expected):
    assert hashlib.sha256(msg).hexdigest() == expected


@pytest.mark.parametrize("msg,expected", NIST.items())
def test_numpy_64_round_nist(msg, expected):
    arr = np.frombuffer(msg, dtype=np.uint8)[None, :]
    assert sha256_numpy(arr, 64)[0].tobytes().hex() == expected


def test_double_sha_genesis():
    # Bitcoin mostra l'hash con i byte invertiti
    assert double_sha256(GENESIS)[::-1].hex() == GENESIS_HASH


def test_numpy_64_coincide_con_hashlib():
    rng = np.random.default_rng(0)
    prefix = os.urandom(76)
    nonces = rng.integers(0, 2**63, size=500, dtype=np.uint64)
    assert np.array_equal(top64_numpy(prefix, nonces, 64), top64_hashlib(prefix, nonces))
    # e il punto d'accesso a 64 round passa da hashlib
    assert np.array_equal(top64(prefix, nonces, 64), top64_hashlib(prefix, nonces))


def test_top64_coincide_con_header_hash():
    prefix = bytes(range(76))
    nonces = np.array([0, 1, 2**32, 2**64 - 1], dtype=np.uint64)
    t = top64_hashlib(prefix, nonces)
    for n, v in zip(nonces.tolist(), t.tolist()):
        assert int.from_bytes(header_hash(prefix, n)[:8], "big") == v


def test_successo_top64_equivale_alla_definizione():
    # Confronto la scorciatoia top64 con la definizione letterale su molti hash
    prefix = os.urandom(76)
    nonces = np.arange(20000, dtype=np.uint64)
    t = top64_hashlib(prefix, nonces)
    for k in (1, 4, 8, 12):
        fast = success_from_top64(t, k)
        slow = np.array([is_success(header_hash(prefix, n), k) for n in nonces.tolist()])
        assert np.array_equal(fast, slow)


def test_round_ridotti_cambiano_hash():
    prefix = os.urandom(76)
    nonces = np.arange(100, dtype=np.uint64)
    assert not np.array_equal(top64_numpy(prefix, nonces, 20), top64_hashlib(prefix, nonces))
