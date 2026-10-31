import hashlib
import os

import numpy as np
import pytest

from config import Config
from dataset import null_rate, oracle_top64, oracle_top64_many, verify
from hashing import is_success_scrypt, scrypt_hash, success_from_top64, top64_scrypt, \
    top64_scrypt_many

CFG = Config(oracolo="scrypt", workers=1)

# RFC 7914, sezione 12: scrypt(P="password", S="NaCl", N=1024, r=8, p=16, dkLen=64)
RFC7914 = ("fdbabe1c9d3472007856e7190d01e9fe7c6ad7cbc8237830e77376634b3731622eaf30d92e22a388"
           "6ff109279d9830dac727afb94a83ee6d8360cbdfa2cc0640")


def test_hashlib_scrypt_vettore_rfc7914():
    out = hashlib.scrypt(b"password", salt=b"NaCl", n=1024, r=8, p=16, dklen=64, maxmem=2**26)
    assert out.hex() == RFC7914


def test_top64_scrypt_coincide_con_la_definizione_letterale():
    prefix = os.urandom(76)
    nonces = np.arange(200, dtype=np.uint64)
    tops = top64_scrypt(prefix, nonces)
    for k in (1, 3, 6):
        atteso = [int.from_bytes(hashlib.scrypt(prefix + int(n).to_bytes(8, "little"),
                                                salt=prefix + int(n).to_bytes(8, "little"),
                                                n=1024, r=1, p=1, dklen=32), "little")
                  < (1 << (256 - k)) for n in nonces]
        assert success_from_top64(tops, k).tolist() == atteso
    d = scrypt_hash(prefix + bytes(8))
    assert is_success_scrypt(d, 3) == bool(success_from_top64(top64_scrypt(prefix, [0]), 3)[0])


def test_scrypt_many_parallelo_uguale_al_seriale():
    pre = [os.urandom(76) for _ in range(5)]
    non = [np.arange(i, i + 2500, dtype=np.uint64) for i in range(5)]  # >2048: spezza in blocchi
    par = top64_scrypt_many(pre, non, workers=2)
    for p, n, t in zip(pre, non, par):
        assert np.array_equal(t, top64_scrypt(p, n))


def test_oracolo_e_verify_coerenti_con_scrypt_reale():
    prefix = os.urandom(76)
    n = np.arange(40, dtype=np.uint64)
    assert np.array_equal(oracle_top64("scrypt", prefix, n), top64_scrypt(prefix, n))
    assert np.array_equal(verify(CFG, prefix, n), top64_scrypt(prefix, n))
    assert np.array_equal(oracle_top64_many("scrypt", [prefix], [n], workers=1)[0],
                          top64_scrypt(prefix, n))


def test_null_rate_scrypt_e_oracolo_sconosciuto():
    assert null_rate("scrypt", 8) == 2.0 ** -8
    with pytest.raises(ValueError, match="oracolo sconosciuto"):
        oracle_top64_many("md5", [b"x" * 76], [np.arange(3, dtype=np.uint64)], workers=1)
    with pytest.raises(ValueError):
        null_rate("md5", 4)


def test_tasso_empirico_scrypt_vicino_a_2_alla_meno_k():
    rng = np.random.default_rng(1)
    pre = [rng.bytes(76) for _ in range(4)]
    non = [np.arange(1500, dtype=np.uint64)] * 4  # 6000 hash, attesi 750 successi a k=3
    y = success_from_top64(np.concatenate(top64_scrypt_many(pre, non, workers=2)), 3)
    assert abs(y.mean() - null_rate("scrypt", 3)) < 0.02
