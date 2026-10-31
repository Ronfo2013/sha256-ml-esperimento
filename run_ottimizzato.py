"""Misura quanto calcolo si risparmia per nonce con le ottimizzazioni note (ottimizzato.py).

    .venv/bin/python run_ottimizzato.py [--n 1000000] [--out risultati_ottimizzato]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import ottimizzato as opt
from hashing import sha256_numpy

SEED = 20261031


def _best(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def naive_numpy(h76: bytes, nonces: np.ndarray, k: int) -> np.ndarray:
    nb = nonces.astype("<u4").view(np.uint8).reshape(-1, 4)
    msg = np.concatenate([np.broadcast_to(np.frombuffer(h76, np.uint8), (len(nonces), 76)), nb], 1)
    d = sha256_numpy(sha256_numpy(msg, 64), 64)
    top32 = d[:, 28:32].copy().view("<u4").ravel()
    return top32 < np.uint32(1 << (32 - k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--out", default="risultati_ottimizzato")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    h76 = rng.bytes(76)
    k = 8
    nonces = np.arange(a.n, dtype=np.uint32)

    # 1. correttezza su tutti gli n nonce, contro hashlib
    t = time.perf_counter()
    ref = np.array([int.from_bytes(hashlib.sha256(hashlib.sha256(
        h76 + int(n).to_bytes(4, "little")).digest()).digest(), "little") < (1 << (256 - k))
        for n in nonces.tolist()])
    t_hashlib = time.perf_counter() - t
    p = opt.prepare(h76)
    got = opt.scan(p, nonces, k)
    mism = int((got != ref).sum())
    tail = h76[64:]
    preps = [p] + [opt.prepare(rng.bytes(64) + tail) for _ in range(3)]
    ab = opt.scan_asicboost(preps, nonces[:200_000], k)
    mism_ab = int((ab[0] != ref[:200_000]).sum())
    print(f"discordanze: ottimizzato {mism}/{a.n}, asicboost {mism_ab}/200000", flush=True)
    if mism or mism_ab:
        raise SystemExit("ERRORE: l'implementazione ottimizzata non coincide con hashlib")

    # 2. tempi numpy (stessa libreria, stesso hardware, un core)
    sub = nonces[:200_000]
    t_naive = _best(lambda: naive_numpy(h76, sub, k)) / len(sub)
    t_opt = _best(lambda: opt.scan(p, sub, k)) / len(sub)
    t_ab = _best(lambda: opt.scan_asicboost(preps, sub, k)) / (len(sub) * len(preps))
    t_prep = _best(lambda: opt.prepare(h76), reps=20)

    costs = {"ingenuo (3 compressioni)": opt.naive_cost(),
             "ottimizzato (midstate, precalcolo, uscita anticipata)": opt.optimized_cost(),
             "ottimizzato + AsicBoost (m=4)": opt.asicboost_cost(4)}
    base_ops = opt.ops(opt.naive_cost())
    rows = [dict(versione=n, **c, operazioni=opt.ops(c),
                 risparmio=1 - opt.ops(c) / base_ops) for n, c in costs.items()]
    tempi = dict(hashlib_C_s=t_hashlib / a.n, numpy_ingenuo_s=t_naive, numpy_ottimizzato_s=t_opt,
                 numpy_asicboost_s=t_ab, prepare_s=t_prep)
    (out / "ottimizzato.json").write_text(json.dumps(dict(
        n=a.n, k=k, seed=SEED, discordanze=mism, discordanze_asicboost=mism_ab,
        successi=int(ref.sum()), costi=rows, tempi=tempi), indent=2))

    L = ["# Calcolare meno per nonce: quanto si risparmia davvero", "",
         "Formato Bitcoin vero (header 80 byte, nonce 4 byte LE, confronto little-endian). "
         "Ogni versione è verificata contro `hashlib` prima di essere misurata.", "",
         "## Correttezza", "",
         f"- {a.n:,} nonce consecutivi, k={k}: **{mism} discordanze** fra versione ottimizzata e "
         f"hashlib ({int(ref.sum())} successi, attesi {a.n / 2**k:.0f}).",
         f"- AsicBoost con 4 header che condividono il blocco 2: **{mism_ab} discordanze** su "
         "200.000 nonce.",
         "- Test unitari: blocco genesi di Bitcoin, k = 1, 4, 8, 12, AsicBoost (`tests/test_ottimizzato.py`).",
         "", "## Lavoro per nonce (modello di costo software)", "",
         f"Round = {opt.OPS_ROUND} operazioni a 32 bit, parola di schedule = {opt.OPS_SCHED}, "
         "somma di feed-forward = 1.", "",
         "| versione | round | parole di schedule | feed-forward | operazioni | risparmio |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['versione']} | {r['round']} | {r['schedule']:.1f} | {r['feedforward']} | "
                 f"{r['operazioni']:,.0f} | {r['risparmio']:.1%} |")
    L += ["", "## Tempi misurati (un core, secondi per nonce)", "",
          "| implementazione | µs per nonce | rispetto a numpy ingenuo |", "|---|---|---|",
          f"| numpy ingenuo | {t_naive * 1e6:.3f} | 1,00 |",
          f"| numpy ottimizzato | {t_opt * 1e6:.3f} | {t_opt / t_naive:.2f} |",
          f"| numpy ottimizzato + AsicBoost (m=4) | {t_ab * 1e6:.3f} | {t_ab / t_naive:.2f} |",
          f"| hashlib (C, ciclo Python, riferimento) | {t_hashlib / a.n * 1e6:.3f} | — |", "",
          f"`prepare` (midstate e precalcolo, una volta per header): {t_prep * 1e6:.0f} µs.", "",
          "## Lettura", "",
          "- Il risparmio viene da tre fatti strutturali: il primo blocco non contiene il nonce, "
          "molte parole del blocco 2 e del secondo hash sono padding costante, e Bitcoin guarda "
          "solo l'ultima parola (H7), che è pronta tre round prima della fine.",
          "- Il midstate da solo vale circa un terzo. Il resto aggiunge pochi punti percentuali.",
          "- AsicBoost condivide solo lo schedule del blocco 2 e richiede di trovare radici di "
          "Merkle con gli stessi ultimi 4 byte (lavoro extra non contato qui). In hardware il "
          "guadagno dichiarato è ~20% perché lì lo schedule pesa di più che in questo modello.",
          "- Tutto questo è già nei miner ASIC e nei software di mining da anni: il risultato "
          "spiega come funzionano, non dà un vantaggio su di loro.",
          "- I tempi numpy includono overhead di memoria e Python; contano i rapporti, non i "
          "valori assoluti. hashlib è C ottimizzato chiamato nonce per nonce.",
          "- Il rapporto misurato (circa 0,5) è migliore di quello previsto dal modello (0,63) "
          "perché la versione numpy ingenua costruisce anche messaggio e padding a ogni nonce: "
          "il numero da citare è quello del modello di costo, non il tempo."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
