"""La sola «soluzione» reale: calcolare MENO per ogni nonce, non indovinarlo.

Formato Bitcoin vero: header di 80 byte, nonce a 4 byte little-endian (byte 76-79).
Successo: doppio SHA-256 letto come intero LITTLE-endian < 2^(256-k).

Tecniche (tutte note e già usate dagli ASIC), ognuna verificata contro hashlib nei test:
1. midstate: il blocco 1 (byte 0-63) non contiene il nonce -> compresso una volta sola;
2. round 0-2 del blocco 2 precalcolati (W0..W2 = coda Merkle, ora, difficoltà) e parte
   costante del round 3 (T1 = costante + W3);
3. schedule precalcolato: nel blocco 2 W4..W15 sono padding fisso, quindi W16 e W17 sono
   costanti e W18, W19 hanno una parte costante;
4. uscita anticipata: H7 del secondo hash = IV7 + registro e dopo il round 60, quindi bastano
   61 round su 64; se H7 non soddisfa la soglia (k <= 32) il nonce si scarta;
5. AsicBoost: m header con midstate diversi ma stesso blocco 2 (stessi ultimi 4 byte della
   radice di Merkle) condividono l'espansione dello schedule del blocco 2.

Modello di costo (operazioni a 32 bit, contate a mano qui sotto): round = 38, parola di
schedule = 21, somma del feed-forward = 1. È un modello software; in hardware i pesi cambiano.
"""
from __future__ import annotations

import numpy as np

from hashing import _IV, _K, _compress, _rotr

OPS_ROUND = 38      # Σ1 11, Ch 4, T1 4 somme, Σ0 11, Maj 5, T2 1, a 1, e 1
OPS_SCHED = 21      # σ0 9, σ1 9, 3 somme
PAD2 = 0x80000000   # primo word di padding


def _s0(x):
    return _rotr(x, 7) ^ _rotr(x, 18) ^ (x >> np.uint32(3))


def _s1(x):
    return _rotr(x, 17) ^ _rotr(x, 19) ^ (x >> np.uint32(10))


def _round(regs, t, wt):
    a, b, c, d, e, f, g, h = regs
    t1 = h + (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) + ((e & f) ^ (~e & g)) + _K[t] + wt
    t2 = (_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))
    return (t1 + t2, a, b, c, d + t1, e, f, g)


def _expand(w, upto):
    for t in range(len(w), upto):
        w.append(w[t - 16] + _s0(w[t - 15]) + w[t - 7] + _s1(w[t - 2]))
    return w


def _u32(x):
    return np.uint32(x)


def _words(b: bytes) -> list:
    return [np.uint32(v) for v in np.frombuffer(b, dtype=">u4")]


# ------------------------------------------------------------------ riferimento ingenuo


def naive_cost() -> dict:
    """3 compressioni complete (blocco 1, blocco 2, secondo hash), ognuna 64 round,
    48 parole di schedule, 8 somme di feed-forward."""
    return dict(round=192, schedule=144, feedforward=24)


# ------------------------------------------------------------------ versione ottimizzata


def prepare(header76: bytes) -> dict:
    """Tutto ciò che non dipende dal nonce: si calcola una volta per header."""
    if len(header76) != 76:
        raise ValueError("servono i primi 76 byte dell'header")
    with np.errstate(over="ignore"):
        b1 = np.array(_words(header76[:64]), dtype=np.uint32)[:, None]
        mid = _compress(_IV[:, None].copy(), b1, 64)[:, 0]
        w0, w1, w2 = _words(header76[64:76])
        regs = tuple(mid)
        for t, wt in enumerate((w0, w1, w2)):
            regs = _round(regs, t, wt)
        a, b, c, d, e, f, g, h = regs
        c1 = h + (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) + ((e & f) ^ (~e & g)) + _K[3]
        c2 = (_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))
        pad = [_u32(PAD2)] + [_u32(0)] * 10 + [_u32(640)]          # W4..W15
        w16 = w0 + _s0(w1) + pad[9 - 4] + _s1(pad[14 - 4])
        w17 = w1 + _s0(w2) + pad[10 - 4] + _s1(pad[15 - 4])
        w18c = w2 + pad[11 - 4] + _s1(w16)                          # + σ0(W3)
        w19c = _s0(pad[4 - 4]) + pad[12 - 4] + _s1(w17)             # + W3
    return dict(mid=mid, regs3=regs, c1=c1, c2=c2, w012=(w0, w1, w2), pad=pad,
                w16=w16, w17=w17, w18c=w18c, w19c=w19c)


def _block2_schedule(p: dict, w3):
    w0, w1, w2 = (np.broadcast_to(x, w3.shape) for x in p["w012"])
    w = [w0, w1, w2, w3] + [np.broadcast_to(x, w3.shape) for x in p["pad"]]
    w += [np.broadcast_to(p["w16"], w3.shape), np.broadcast_to(p["w17"], w3.shape),
          p["w18c"] + _s0(w3), p["w19c"] + w3]
    return _expand(w, 64)


def _block2_rounds(p: dict, mid, regs3, c1, c2, w, w3):
    a, b, c, d, e, f, g, h = regs3
    t1 = c1 + w3                                    # round 3 con la parte costante
    regs = (t1 + c2, a, b, c, d + t1, e, f, g)
    for t in range(4, 64):
        regs = _round(regs, t, w[t])
    return [mid[i] + regs[i] for i in range(8)]     # feed-forward: stato dopo il blocco 2


def _second_hash_h7(h1):
    w = list(h1) + [np.broadcast_to(_u32(x), h1[0].shape)
                    for x in (PAD2, 0, 0, 0, 0, 0, 0, 256)]
    w = _expand(w, 61)                              # W16..W60: servono solo fino al round 60
    regs = tuple(np.broadcast_to(x, h1[0].shape) for x in _IV)
    for t in range(61):                             # uscita anticipata: 61 round su 64
        regs = _round(regs, t, w[t])
    return _IV[7] + regs[4]                         # H7 = IV7 + e dopo il round 60


def h7_to_top32(h7):
    """I 4 byte più significativi dell'intero little-endian = H7 con i byte invertiti."""
    return h7.byteswap()


def scan(p: dict, nonces: np.ndarray, k: int) -> np.ndarray:
    """Successo (bool) per ogni nonce uint32. Esatto per k <= 32."""
    if not 1 <= k <= 32:
        raise ValueError("uscita anticipata esatta solo per k <= 32")
    with np.errstate(over="ignore"):
        w3 = np.asarray(nonces, dtype=np.uint32).byteswap()   # LE in memoria, letto BE
        w = _block2_schedule(p, w3)
        h1 = _block2_rounds(p, p["mid"], p["regs3"], p["c1"], p["c2"], w, w3)
        top32 = h7_to_top32(_second_hash_h7(h1))
    return top32 < np.uint32(1 << (32 - k)) if k < 32 else top32 == 0


def optimized_cost() -> dict:
    # blocco 2: round 3 parziale (conto 1 round intero) + 4..63 = 61; schedule W18..W63 = 46
    # secondo hash: 61 round, schedule W16..W60 = 45, feed-forward 8 (blocco 2) + 1 (H7)
    return dict(round=61 + 61, schedule=46 + 45, feedforward=9)


# ------------------------------------------------------------------ AsicBoost


def scan_asicboost(preps: list[dict], nonces: np.ndarray, k: int) -> list[np.ndarray]:
    """m header con lo STESSO blocco 2 (stessi byte 64-79): lo schedule del blocco 2 si
    espande una volta per nonce e serve a tutti gli m midstate."""
    if len({tuple(int(x) for x in p["w012"]) for p in preps}) != 1:
        raise ValueError("AsicBoost richiede lo stesso blocco 2 (byte 64-75 uguali)")
    with np.errstate(over="ignore"):
        w3 = np.asarray(nonces, dtype=np.uint32).byteswap()
        w = _block2_schedule(preps[0], w3)          # condiviso
        out = []
        for p in preps:
            h1 = _block2_rounds(p, p["mid"], p["regs3"], p["c1"], p["c2"], w, w3)
            top32 = h7_to_top32(_second_hash_h7(h1))
            out.append(top32 < np.uint32(1 << (32 - k)))
    return out


def asicboost_cost(m: int) -> dict:
    c = optimized_cost()
    return dict(round=c["round"], schedule=46 / m + 45, feedforward=c["feedforward"])


def ops(cost: dict) -> float:
    return cost["round"] * OPS_ROUND + cost["schedule"] * OPS_SCHED + cost["feedforward"]
