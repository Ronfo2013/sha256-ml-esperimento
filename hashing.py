"""Funzione obiettivo: doppio SHA-256 di (prefisso + nonce a 8 byte little-endian).

Due implementazioni:
- `hashlib` (riferimento, SHA-256 reale a 64 round): verifica di ogni candidato.
- `numpy` vettoriale con numero di round configurabile: serve SOLO al controllo
  positivo (SHA-256 indebolito). A 64 round deve coincidere con hashlib (test).

Per decidere il successo basta il "top64": i primi 8 byte dell'hash letti big-endian.
hash < 2^(256-k)  <=>  i primi k bit sono zero  <=>  top64 < 2^(64-k)   (per k <= 64)
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

NONCE_BYTES = 8


def nonce_to_bytes(nonce: int) -> bytes:
    return int(nonce).to_bytes(NONCE_BYTES, "little")


def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def header_hash(prefix: bytes, nonce: int) -> bytes:
    return double_sha256(prefix + nonce_to_bytes(nonce))


def is_success(digest: bytes, k: int) -> bool:
    """Definizione letterale: interpretazione big-endian dell'hash < 1 << (256 - k)."""
    return int.from_bytes(digest, "big") < (1 << (256 - k))


def success_from_top64(top64: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k <= 64:
        raise ValueError("k deve stare fra 1 e 64")
    return top64 < np.uint64(1 << (64 - k))


# ---------------------------------------------------------------- hashlib, in blocco

def _top64_hashlib_chunk(args) -> np.ndarray:
    prefix, nonces, double = args
    base = hashlib.sha256(prefix)  # stato dopo il prefisso, riusato per ogni nonce
    sha = hashlib.sha256
    out = np.empty(len(nonces), dtype=np.uint64)
    for i, n in enumerate(nonces.tolist()):
        h = base.copy()
        h.update(n.to_bytes(NONCE_BYTES, "little"))
        d = sha(h.digest()).digest() if double else h.digest()
        out[i] = int.from_bytes(d[:8], "big")
    return out


def top64_hashlib(prefix: bytes, nonces: np.ndarray, double: bool = True) -> np.ndarray:
    return _top64_hashlib_chunk((prefix, np.asarray(nonces, dtype=np.uint64), double))


# ---------------------------------------------------------------- scrypt (Litecoin/Dogecoin)

SCRYPT_PARAMS = dict(n=1024, r=1, p=1, dklen=32)


def scrypt_hash(msg: bytes) -> bytes:
    """scrypt come nel proof-of-work di Litecoin: password = salt = messaggio, N=1024, r=1, p=1."""
    return hashlib.scrypt(msg, salt=msg, **SCRYPT_PARAMS)


def is_success_scrypt(digest: bytes, k: int) -> bool:
    """Definizione letterale (Litecoin): digest come intero little-endian < 1 << (256 - k)."""
    return int.from_bytes(digest, "little") < (1 << (256 - k))


def _top64_scrypt_chunk(args) -> np.ndarray:
    prefix, nonces = args
    out = np.empty(len(nonces), dtype=np.uint64)
    for i, n in enumerate(nonces.tolist()):
        out[i] = int.from_bytes(scrypt_hash(prefix + n.to_bytes(NONCE_BYTES, "little"))[24:32],
                                "little")
    return out


def top64_scrypt(prefix: bytes, nonces: np.ndarray) -> np.ndarray:
    """Top64 per scrypt: i 8 byte ALTI del confronto little-endian sono digest[24:32]."""
    return _top64_scrypt_chunk((prefix, np.asarray(nonces, dtype=np.uint64)))


def _top64_scrypt_job(args) -> np.ndarray:
    return _top64_scrypt_chunk(args)


def top64_scrypt_many(prefixes: list[bytes], nonces_per_prefix: list[np.ndarray],
                      workers: int | None = None) -> list[np.ndarray]:
    """Come `top64_many`, ma per scrypt. Un job = un prefisso, spezzato in blocchi da 2048
    nonce perché ogni hash costa ~250 µs e i prefissi possono essere pochi."""
    jobs, spans = [], []
    for p, n in zip(prefixes, nonces_per_prefix):
        n = np.asarray(n, dtype=np.uint64)
        start = len(jobs)
        jobs += [(p, n[i:i + 2048]) for i in range(0, len(n), 2048)]
        spans.append((start, len(jobs)))
    if workers == 1 or len(jobs) < 4:
        res = [_top64_scrypt_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_top64_scrypt_job, jobs, chunksize=max(1, len(jobs) // 64)))
    return [np.concatenate(res[a:b]) if b > a else np.empty(0, dtype=np.uint64)
            for a, b in spans]


# ---------------------------------------------------------------- numpy, round ridotti

_K = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
], dtype=np.uint32)
_IV = np.array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19], dtype=np.uint32)


def _rotr(x, r):
    return (x >> np.uint32(r)) | (x << np.uint32(32 - r))


def _compress(state: np.ndarray, block: np.ndarray, rounds: int) -> np.ndarray:
    """state (8, n) uint32, block (16, n) uint32 -> nuovo stato (8, n), con feed-forward."""
    w = list(block)
    for t in range(16, rounds):
        s0 = _rotr(w[t - 15], 7) ^ _rotr(w[t - 15], 18) ^ (w[t - 15] >> np.uint32(3))
        s1 = _rotr(w[t - 2], 17) ^ _rotr(w[t - 2], 19) ^ (w[t - 2] >> np.uint32(10))
        w.append(w[t - 16] + s0 + w[t - 7] + s1)
    a, b, c, d, e, f, g, h = (state[i].copy() for i in range(8))
    for t in range(rounds):
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = h + S1 + ch + _K[t] + w[t]
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = S0 + maj
        h, g, f, e, d, c, b, a = g, f, e, d + t1, c, b, a, t1 + t2
    return state + np.stack([a, b, c, d, e, f, g, h])


def sha256_numpy(messages: np.ndarray, rounds: int = 64) -> np.ndarray:
    """messages: (n, L) uint8, tutti della stessa lunghezza. Ritorna (n, 32) uint8."""
    if not 1 <= rounds <= 64:
        raise ValueError("rounds deve stare fra 1 e 64")
    messages = np.ascontiguousarray(messages, dtype=np.uint8)
    n, length = messages.shape
    pad_len = (55 - length) % 64
    padding = np.zeros((n, 1 + pad_len + 8), dtype=np.uint8)
    padding[:, 0] = 0x80
    padding[:, -8:] = np.frombuffer((length * 8).to_bytes(8, "big"), dtype=np.uint8)
    full = np.concatenate([messages, padding], axis=1)
    words = full.view(">u4").astype(np.uint32).T  # (parole, n)
    state = np.repeat(_IV[:, None], n, axis=1)
    with np.errstate(over="ignore"):
        for start in range(0, words.shape[0], 16):
            state = _compress(state, words[start:start + 16], rounds)
    return np.ascontiguousarray(state.T.astype(">u4")).view(np.uint8).reshape(n, 32)


def build_messages(prefix: bytes, nonces: np.ndarray) -> np.ndarray:
    nonces = np.asarray(nonces, dtype=np.uint64)
    nb = nonces.astype("<u8").view(np.uint8).reshape(-1, NONCE_BYTES)
    pre = np.broadcast_to(np.frombuffer(prefix, dtype=np.uint8), (len(nonces), len(prefix)))
    return np.concatenate([pre, nb], axis=1)


def top64_numpy(prefix: bytes, nonces: np.ndarray, rounds: int, double: bool = True) -> np.ndarray:
    """SHA-256 a `rounds` round (stessi round in ogni compressione), singolo o doppio."""
    digest = sha256_numpy(build_messages(prefix, nonces), rounds)
    if double:
        digest = sha256_numpy(digest, rounds)
    return digest[:, :8].copy().view(">u8").astype(np.uint64).ravel()


def top64(prefix: bytes, nonces: np.ndarray, rounds: int = 64, double: bool = True) -> np.ndarray:
    """Punto unico d'accesso: a 64 round usa sempre hashlib (SHA-256 reale)."""
    if rounds == 64:
        return top64_hashlib(prefix, nonces, double)
    return top64_numpy(prefix, nonces, rounds, double)


def _top64_job(args) -> np.ndarray:
    return top64(*args)


def top64_many(prefixes: list[bytes], nonces_per_prefix: list[np.ndarray], rounds: int = 64,
               double: bool = True, workers: int | None = None) -> list[np.ndarray]:
    """Stesso calcolo su più prefissi, in parallelo su più processi."""
    jobs = [(p, np.asarray(n, dtype=np.uint64), rounds, double)
            for p, n in zip(prefixes, nonces_per_prefix)]
    if workers == 1 or len(jobs) < 4:
        return [_top64_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_top64_job, jobs, chunksize=max(1, len(jobs) // 64)))
