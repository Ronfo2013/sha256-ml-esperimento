"""Valutazione sui prefissi di test: selettore top-M, metriche, intervalli e test statistici.

Unità statistica indipendente = prefisso. Il bootstrap ricampiona i prefissi (a grappolo),
non i singoli hash, perché gli hash dello stesso prefisso condividono il modello e il pool.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from baselines import first_hit, random_search, sequential
from config import Config
from dataset import features, null_rate, verify
from hashing import success_from_top64, top64_hashlib, top64_scrypt


@dataclass
class MethodResult:
    name: str
    successes: np.ndarray            # per prefisso, fra gli M verificati
    attempts: list                   # tentativi al primo successo per prefisso (None = non trovato)
    verified: int                    # M per prefisso
    extra: dict = field(default_factory=dict)

    @property
    def total_success(self) -> int:
        return int(self.successes.sum())

    @property
    def total_verified(self) -> int:
        return self.verified * len(self.successes)

    @property
    def rate(self) -> float:
        return self.total_success / self.total_verified


def bootstrap_rate_ci(successes: np.ndarray, per_prefix: int, reps: int, seed,
                      alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(successes), size=(reps, len(successes)))
    rates = successes[idx].sum(axis=1) / (per_prefix * len(successes))
    return float(np.quantile(rates, alpha / 2)), float(np.quantile(rates, 1 - alpha / 2))


def bootstrap_ratio_ci(a: np.ndarray, b: np.ndarray, reps: int, seed, alpha=0.05):
    """IC del rapporto (lift) fra due metodi valutati sugli stessi prefissi (appaiati)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
    ratio = np.where(sb > 0, sa / np.maximum(sb, 1), np.nan)
    ratio = ratio[~np.isnan(ratio)]
    if ratio.size < reps / 2:  # troppi ricampionamenti senza successi nel riferimento: IC non definito
        return float("nan"), float("nan")
    return float(np.quantile(ratio, alpha / 2)), float(np.quantile(ratio, 1 - alpha / 2))


def binom_greater(k: int, n: int, p0: float) -> float:
    return float(stats.binomtest(k, n, p0, alternative="greater").pvalue)


def ztest_two_prop(k1, n1, k2, n2) -> float:
    """z-test unilaterale: il metodo 1 ha tasso maggiore del metodo 2?"""
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return float(stats.norm.sf(z))


def min_detectable_lift(n: int, p0: float, alpha: float, power: float = 0.8) -> float:
    """Miglioramento relativo minimo rilevabile con n hash verificati (approssimazione normale)."""
    za, zb = stats.norm.isf(alpha), stats.norm.isf(1 - power)
    # risolve p1: (p1 - p0) * sqrt(n) = za*sqrt(p0 q0) + zb*sqrt(p1 q1), iterando
    p1 = p0 * 1.5
    for _ in range(50):
        p1 = p0 + (za * np.sqrt(p0 * (1 - p0)) + zb * np.sqrt(p1 * (1 - p1))) / np.sqrt(n)
    return p1 / p0 - 1


def auc_hanley(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """AUC con errore standard di Hanley-McNeil (1982)."""
    if y.min() == y.max():
        return float("nan"), float("nan")
    a = roc_auc_score(y, s)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    return float(a), float(se)


def hash_seconds_per_candidate(prefix: bytes, n: int = 200_000, oracolo: str = "sha256d") -> float:
    """Secondi per un hash dell'oracolo, su un core (unità dei «hash equivalenti»)."""
    if oracolo == "scrypt":
        n = min(n, 2000)
    nonces = np.arange(n, dtype=np.uint64)
    t = time.perf_counter()
    top64_scrypt(prefix, nonces) if oracolo == "scrypt" else top64_hashlib(prefix, nonces)
    return (time.perf_counter() - t) / n


class TestBench:
    """Pool di candidati per prefisso di test, con hash calcolati UNA volta per i riferimenti
    (AUC, baseline casuale). Le scelte del modello vengono poi RI-verificate a parte."""

    def __init__(self, cfg: Config, prefixes, pools, pool_tops, seq_tops):
        self.cfg, self.prefixes, self.pools = cfg, prefixes, pools
        self.pool_success = [success_from_top64(t, cfg.k) for t in pool_tops]
        self.pool_success_train = [success_from_top64(t, cfg.k_train) for t in pool_tops]
        self.seq_success = [success_from_top64(t, cfg.k) for t in seq_tops]

    def baseline_sequential(self) -> MethodResult:
        m = self.cfg.top_m
        res = [sequential(s, m) for s in self.seq_success]
        return MethodResult("sequenziale", np.array([r[0] for r in res]), [r[1] for r in res], m)

    def baseline_random(self, seed: int) -> MethodResult:
        m = self.cfg.top_m
        res = [random_search(s, m, [self.cfg.master_seed, 5, seed, i])
               for i, s in enumerate(self.pool_success)]
        return MethodResult(f"casuale (seed {seed})", np.array([r[0] for r in res]),
                            [r[1] for r in res], m)

    def run_model(self, scorer, label: str) -> MethodResult:
        cfg, m = self.cfg, self.cfg.top_m
        succ, attempts, scores_all, ytrue_k, ytrue_kt = [], [], [], [], []
        mismatches = 0
        for i, (prefix, pool) in enumerate(zip(self.prefixes, self.pools)):
            X = features(prefix, pool, cfg)
            order = scorer.rank(X, [cfg.master_seed, 6, scorer.seed, i])
            chosen = pool[order[:m]]
            # verifica REALE e indipendente dei candidati scelti
            verified = success_from_top64(verify(cfg, prefix, chosen), cfg.k)
            mismatches += int((verified != self.pool_success[i][order[:m]]).sum())
            succ.append(int(verified.sum()))
            attempts.append(first_hit(self.pool_success[i][order]))
            # punteggio per AUC: rango normalizzato dentro il prefisso
            ranks = np.empty(len(order))
            ranks[order] = np.arange(len(order), 0, -1) / len(order)
            scores_all.append(ranks)
            ytrue_k.append(self.pool_success[i])
            ytrue_kt.append(self.pool_success_train[i])
        if mismatches:
            raise RuntimeError(f"verifica incoerente su {mismatches} candidati")
        s = np.concatenate(scores_all)
        y_k = np.concatenate(ytrue_k).astype(int)
        y_kt = np.concatenate(ytrue_kt).astype(int)
        auc_k, se_k = auc_hanley(y_k, s)
        auc_kt, se_kt = auc_hanley(y_kt, s)
        top_frac = m / cfg.n_candidates
        recall = float(np.sum(succ) / max(1, y_k.sum()))
        return MethodResult(label, np.array(succ), attempts, m, extra=dict(
            auc_k=auc_k, auc_k_se=se_k, auc_ktrain=auc_kt, auc_ktrain_se=se_kt,
            precision=float(np.sum(succ) / (m * len(succ))), recall=recall,
            recall_random=top_frac, train_s=scorer.train_seconds,
            s_per_candidate=scorer.seconds_per_candidate))


def compare(cfg: Config, method: MethodResult, random_ref: MethodResult, n_tests: int,
            seed: int) -> dict:
    """Test contro il caso. Correzione di Bonferroni sul numero di confronti della corsa."""
    p0 = null_rate(cfg.oracolo, cfg.k)
    alpha_corr = cfg.alpha / max(1, n_tests)
    p_binom = binom_greater(method.total_success, method.total_verified, p0)
    p_z = ztest_two_prop(method.total_success, method.total_verified,
                         random_ref.total_success, random_ref.total_verified)
    lo, hi = bootstrap_rate_ci(method.successes, method.verified, cfg.bootstrap_reps,
                               [cfg.master_seed, 7, seed], cfg.alpha)
    llo, lhi = bootstrap_ratio_ci(method.successes, random_ref.successes, cfg.bootstrap_reps,
                                  [cfg.master_seed, 8, seed], cfg.alpha)
    found = [a for a in method.attempts if a is not None]
    # il modello ordina meglio del caso? AUC alla soglia di addestramento (molti più positivi)
    auc_kt, se_kt = method.extra.get("auc_ktrain"), method.extra.get("auc_ktrain_se")
    p_auc = float(stats.norm.sf((auc_kt - 0.5) / se_kt)) if se_kt else float("nan")
    return dict(
        p_auc_ktrain=p_auc, impara=bool(p_auc < alpha_corr),
        metodo=method.name, successi=method.total_success, verificati=method.total_verified,
        tasso=method.rate, ic_basso=lo, ic_alto=hi, tasso_nullo=p0,
        lift_vs_teorico=method.rate / p0,
        lift_vs_casuale=(method.rate / random_ref.rate if random_ref.rate else float("nan")), lift_ic=(llo, lhi),
        tentativi_per_successo=(method.total_verified / method.total_success
                                if method.total_success else float("inf")),
        tentativi_primo_successo_medio=float(np.mean(found)) if found else float("nan"),
        prefissi_senza_successo=sum(a is None for a in method.attempts),
        p_binomiale=p_binom, p_ztest=p_z, alpha_corretto=alpha_corr,
        significativo=bool(p_binom < alpha_corr and p_z < alpha_corr),
        **method.extra)
