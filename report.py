"""Report Markdown + grafici PNG a partire da risultati.json."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODEL_ORDER = ["maggioritaria", "logistica", "gbm", "mlp", "gbm_mescolato"]
COL = {"sequenziale": "#8c8c8c", "casuale": "#4c78a8", "maggioritaria": "#bab0ac",
       "logistica": "#f58518", "gbm": "#54a24b", "mlp": "#b279a2", "gbm_mescolato": "#e45756"}


def _fmt_p(p):
    return "<1e-300" if p == 0 else f"{p:.2g}"


def _key(name: str) -> str:
    return name.split(" (")[0]


def _costo(r, run, cfg):
    """Hash equivalenti spesi per ogni successo trovato, contando inferenza e addestramento."""
    if not r["successi"]:
        return float("inf")
    cand = cfg["n_candidates"] * cfg["n_test_prefixes"]
    inf = cand * r.get("s_per_candidate", 0) / run["hash_s"]
    tr = r.get("train_s", 0) / run["hash_s"]
    return (r["verificati"] + inf + tr) / r["successi"]


def fig_rates(runs, out: Path):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.2), squeeze=False)
    for ax, run in zip(axes[0], runs):
        rows = run["baseline"][:2] + [r for r in run["modelli"] if r["seed"] == 0]
        names = [_key(r["metodo"]) for r in rows]
        lift = [r["tasso"] / run["tasso_nullo"] for r in rows]
        lo = [r["tasso"] / run["tasso_nullo"] - r["ic_basso"] / run["tasso_nullo"] for r in rows]
        hi = [r["ic_alto"] / run["tasso_nullo"] - r["tasso"] / run["tasso_nullo"] for r in rows]
        ax.bar(names, lift, yerr=[lo, hi], capsize=4, color=[COL.get(n, "#999") for n in names])
        ax.axhline(1, color="black", lw=1, ls="--")
        ax.set_title(f"{run['oracolo']}, k={run['k']}")
        ax.set_ylabel("tasso di successo / tasso del caso")
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Tasso di successo per hash verificato (IC 95% bootstrap sui prefissi)")
    fig.tight_layout()
    fig.savefig(out / "tassi.png", dpi=130)
    plt.close(fig)


def fig_curve(runs, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for run in runs:
        c = run["curva"]
        x = [r["righe"] for r in c]
        ax.plot(x, [r["auc_val"] for r in c], marker="o", label=f"{run['oracolo']} k={run['k']}")
        ax.plot(x, [r["auc_val_mescolato"] for r in c], ls=":", marker="x",
                label=f"{run['oracolo']} k={run['k']}, etichette mescolate")
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("righe di addestramento")
    ax.set_ylabel("AUC su validazione (prefissi nuovi)")
    ax.set_title("Curva di apprendimento (gradient boosting)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curva_apprendimento.png", dpi=130)
    plt.close(fig)


def fig_attempts(run, out: Path, cap: int):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bins = np.logspace(0, np.log10(cap), 30)
    for name, att in run["tentativi"].items():
        vals = [a for a in att if a is not None]
        ax.hist(vals, bins=bins, histtype="step", lw=1.6, label=f"{name} (mancati: "
                f"{sum(a is None for a in att)})", color=COL.get(_key(name), None))
    ax.axvline(1 / run["tasso_nullo"], color="black", ls="--", lw=1, label="atteso dal caso")
    ax.set_xscale("log")
    ax.set_xlabel("hash calcolati fino al primo successo")
    ax.set_ylabel("prefissi")
    ax.set_title(f"Distribuzione dei tentativi ({run['oracolo']}, k={run['k']})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"tentativi_{run['oracolo']}_k{run['k']}.png", dpi=130)
    plt.close(fig)


def fig_rounds(rows, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for fn, mk in (("singolo", "o"), ("doppio", "s")):
        r = [x for x in rows if x["funzione"] == fn]
        ax.plot([x["round"] for x in r], [x["auc_stesso_prefisso"] for x in r], marker=mk,
                label=f"SHA-256 {fn}, stesso prefisso")
        ax.plot([x["round"] for x in r], [x["auc_prefissi_nuovi"] for x in r], marker=mk, ls=":",
                label=f"SHA-256 {fn}, prefissi nuovi")
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([x["round"] for x in rows if x["funzione"] == "singolo"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("round di SHA-256")
    ax.set_ylabel("AUC (etichetta: primi 4 bit a zero)")
    ax.set_title("Dove si perde la struttura")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "round_ridotti.png", dpi=130)
    plt.close(fig)


def _run_table(run, cfg) -> list[str]:
    L = ["| metodo | successi / verificati | tasso × caso [IC 95%] | lift vs casuale [IC 95%] "
         "| p binomiale | p z-test | significativo | AUC test (k) | AUC test (k_train), p "
         "| AUC val (k_train) | richiamo (caso) | hash per successo "
         "| costo in hash eq. per successo |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    p0 = run["tasso_nullo"]
    for r in run["baseline"] + run["modelli"]:
        is_model = "auc_k" in r
        sig = "—" if r["significativo"] is None else ("**SÌ**" if r["significativo"] else "no")
        auc = f"{r['auc_k']:.3f} ± {r['auc_k_se']:.3f}" if is_model else "—"
        aucv = f"{r['auc_val']:.3f}" if is_model else "—"
        auckt = (f"{r['auc_ktrain']:.4f} ± {r['auc_ktrain_se']:.4f}, p={_fmt_p(r['p_auc_ktrain'])}"
                 + (" **impara**" if r["impara"] else "")) if is_model else "—"
        rec = f"{r['recall']:.3f} ({r['recall_random']:.3f})" if is_model else "—"
        costo = f"{_costo(r, run, cfg):,.0f}" if is_model else f"{r['tentativi_per_successo']:,.0f}"
        L.append(f"| {r['metodo']} | {r['successi']} / {r['verificati']:,} | "
                 f"{r['tasso'] / p0:.3f} [{r['ic_basso'] / p0:.3f}, {r['ic_alto'] / p0:.3f}] | "
                 f"{r['lift_vs_casuale']:.3f} [{r['lift_ic'][0]:.3f}, {r['lift_ic'][1]:.3f}] | "
                 f"{_fmt_p(r['p_binomiale'])} | {_fmt_p(r['p_ztest'])} | {sig} | {auc} | {auckt} | "
                 f"{aucv} | "
                 f"{rec} | {r['tentativi_per_successo']:,.0f} | {costo} |")
    return L


def auc_max_piantato(k_train: int) -> float:
    """AUC massima ottenibile sull'oracolo piantato: il segnale riguarda 1 nonce su 16."""
    p = 2.0 ** -k_train
    pi_pos = 2 / 17
    pi_neg = (1 / 16) * (1 - 2 * p) / (1 - 17 * p / 16)
    return 0.5 + 0.5 * (pi_pos - pi_neg)


CONTROLLI = ("gbm_mescolato", "maggioritaria")


def verdict(runs) -> list[str]:
    L = []
    for run in runs:
        sig = [r for r in run["modelli"] if r["significativo"]]
        real = [r for r in sig if not r["metodo"].startswith(CONTROLLI)]
        learn = [r for r in run["modelli"] if r["impara"] and not r["metodo"].startswith(CONTROLLI)]
        bad = [r for r in run["modelli"] if r["metodo"].startswith(CONTROLLI)
               and (r["significativo"] or r["impara"])]
        tag = f"**{run['oracolo']}, k={run['k']}**"
        if run["oracolo"] == "piantato":
            best = max(run["modelli"], key=lambda r: r["auc_ktrain"])
            amax = auc_max_piantato(run["k_train"])
            if not learn:
                L.append(f"- {tag} (controllo positivo): il segnale piantato NON è stato "
                         "imparato da nessun modello: la pipeline ha un problema e i risultati "
                         "negativi su SHA-256 non vanno considerati validi.")
            else:
                sel = ("e il selettore ottiene un tasso di successo significativamente sopra il "
                       "caso." if real else
                       "ma il test del selettore non è significativo: con questo numero di hash "
                       f"verificati il miglioramento minimo rilevabile è circa "
                       f"+{100 * run['mde']:.0f}%, e il vantaggio piantato è al massimo +88%.")
                L.append(f"- {tag} (controllo positivo): il segnale piantato è stato imparato "
                         f"({len(learn)} modelli; migliore {best['metodo']}, AUC "
                         f"{best['auc_ktrain']:.3f} su un massimo teorico di {amax:.3f}) {sel}")
        elif real or learn:
            L.append(f"- {tag}: {len(real)} modelli significativi sul selettore, {len(learn)} "
                     "sull'AUC. Prima di crederci, cercare leakage o errori: sarebbe un "
                     "risultato straordinario.")
        else:
            L.append(f"- {tag}: **nessun modello batte il caso**: né il selettore né l'AUC sono "
                     f"significativi (Bonferroni su {run['confronti']} confronti). Con questi dati un "
                     f"miglioramento superiore a circa il {100 * run['mde']:.0f}% sarebbe stato "
                     "rilevato con l'80% di probabilità.")
        if bad:
            L.append(f"  - ATTENZIONE: risultano significativi anche dei controlli che dovrebbero "
                     f"stare al caso ({', '.join(r['metodo'] for r in bad)}).")
    return L


def write(data: dict, out: Path):
    runs, cfg, meta = data["corse"], data["meta"]["config"], data["meta"]
    fig_rates(runs, out)
    fig_curve(runs, out)
    for run in runs:
        fig_attempts(run, out, cfg["n_candidates"])
    fig_rounds(data["round"], out)

    L = ["# Esperimento: un modello di ML sceglie nonce SHA-256 migliori del caso?", "",
         f"Comando: `.venv/bin/python {meta['comando']}`  ",
         f"Seed principale: `{cfg['master_seed']}` (tutti gli altri seed derivano da questo); "
         f"seed dei modelli: {cfg['model_seeds']}  ",
         f"Ambiente: Python {meta['python']}, numpy {meta['numpy']}, scikit-learn "
         f"{meta['sklearn']}, {meta['macchina']}. Durata: {meta['secondi_totali']:.0f} s.", "",
         "## 1. Leakage e bias segnalati prima dei risultati", "",
         f"- **Split per prefisso.** Prefissi in comune fra train, validazione e test: "
         f"{sum(r['controlli']['prefissi_condivisi'] for r in runs)}. Il test usa solo prefissi "
         "mai visti in addestramento.",
         "- **Verifica reale.** Ogni candidato scelto da un modello è stato ricalcolato a parte "
         "con `hashlib`; le incoerenze con gli hash del pool sono 0 (altrimenti la corsa si "
         "interrompe).",
         "- **Feature.** Solo bit del nonce, statistiche elementari del nonce e bit della coda "
         "del prefisso: niente che derivi dall'hash.",
         f"- **Soglia di addestramento più facile.** I modelli imparano l'etichetta a "
         f"k_train={cfg['k_train']} (positivi 1 su {2 ** cfg['k_train']}), non a k: con k=12 i "
         "positivi sarebbero troppo pochi. È una scelta di progetto, non leakage: il successo "
         "si conta sempre alla soglia vera k.",
         f"- **Sottocampionamento dei negativi** in addestramento (al massimo "
         f"{cfg['max_train_rows']:,} righe, tutti i positivi tenuti). Le probabilità non sono "
         "calibrate, ma serve solo l'ordinamento.",
         "- **Pareggi.** A parità di punteggio l'ordine è casuale con seed: la classe "
         "maggioritaria equivale quindi alla ricerca casuale, come deve.",
         "- **Iperparametri fissati prima** di guardare il test; nessuna ricerca sul test.",
         "- **Confronti multipli.** Correzione di Bonferroni sul numero di modelli × seed di ogni "
         "corsa. Le corse a k diversi non sono corrette fra loro.",
         "- **Bootstrap a grappolo** sui prefissi: gli hash dello stesso prefisso non sono "
         "osservazioni indipendenti rispetto al modello.",
         "- **z-test.** Confronta il modello con la ricerca casuale nello stesso pool: i due "
         "campioni si sovrappongono un po', quindi il test principale è quello binomiale contro "
         "il tasso teorico 2^-k.",
         "- **Oracolo «piantato».** È una funzione finta, costruita apposta con un vantaggio "
         "(×2 sui nonce con i 4 bit bassi a zero): non è SHA-256 e serve solo a provare la "
         "pipeline.",
         "- **Formato.** Il nonce è di 8 byte little-endian come da specifica; in Bitcoin è di 4 "
         "byte, e il prefisso è casuale invece di un vero header.", "",
         "## 2. Potenza statistica", "",
         "| corsa | hash verificati per metodo | tasso del caso | miglioramento minimo "
         "rilevabile (potenza 80%) |", "|---|---|---|---|"]
    for run in runs:
        L.append(f"| {run['oracolo']}, k={run['k']} | {run['verificati_per_metodo']:,} | "
                 f"{run['tasso_nullo']:.3g} | +{100 * run['mde']:.0f}% |")
    L += ["", "## 3. Esito in breve", ""] + verdict(runs) + ["", "## 4. Risultati per corsa", ""]
    for run in runs:
        L += [f"### {run['oracolo']}, k={run['k']}", "",
              f"Addestramento: {run['righe_train']:,} righe, {run['positivi_train']:,} positivi "
              f"a k_train. Test: {cfg['n_test_prefixes']} prefissi nuovi × {cfg['n_candidates']:,}"
              f" candidati, verificati i primi {cfg['top_m']:,}. Un doppio hash costa "
              f"{run['hash_s'] * 1e6:.2f} µs.", ""]
        L += _run_table(run, cfg)
        L += ["", f"![tentativi](tentativi_{run['oracolo']}_k{run['k']}.png)", ""]
    L += ["## 5. Confronto dei tassi di successo", "", "![tassi](tassi.png)", "",
          "## 6. Curva di apprendimento", "", "![curva](curva_apprendimento.png)", "",
          "## 7. Round ridotti: dove si perde la struttura", "",
          "| funzione | round | AUC stesso prefisso (min–max) | AUC prefissi nuovi |",
          "|---|---|---|---|"]
    for r in data["round"]:
        L.append(f"| {r['funzione']} | {r['round']} | {r['auc_stesso_prefisso']:.3f} "
                 f"({r['auc_stesso_prefisso_min']:.3f}–{r['auc_stesso_prefisso_max']:.3f}) | "
                 f"{r['auc_prefissi_nuovi']:.3f} |")
    L += ["", "Lettura: «stesso prefisso» misura se esiste struttura in assoluto (leakage voluto). "
          "Un'AUC su prefissi nuovi lontana da 0,5 al round 4 del singolo (sopra o sotto, a seconda "
          "del seed) non è un segnale che generalizza: con pochi prefissi di addestramento il "
          "modello impara la finestra di valori buoni di quei prefissi, che sui prefissi nuovi "
          "cade altrove per caso. Con la prova rapida (2 prefissi) era 0,23, con la completa "
          "(3 prefissi) 0,60.", "",
          "![round](round_ridotti.png)", "",
          "## 8. Come leggere le colonne", "",
          "- **tasso × caso**: 1,0 vuol dire uguale al caso; 2,0 vuol dire il doppio dei successi.",
          "- **lift vs casuale**: stesso confronto, ma con la ricerca casuale fatta davvero sugli "
          "stessi candidati.",
          "- **p**: probabilità di vedere un vantaggio almeno così grande se il modello non "
          "servisse a niente. «Significativo» richiede entrambi i p sotto α/numero di confronti.",
          "- **AUC**: 0,5 = il modello ordina a caso; 1,0 = ordina perfettamente.",
          "- **richiamo (caso)**: quota dei successi del pool finiti fra i candidati scelti; fra "
          "parentesi il valore atteso scegliendo a caso.",
          "- **costo in hash eq.**: hash verificati più tempo di inferenza (su tutti gli N "
          "candidati) e di addestramento, convertiti in hash equivalenti, diviso i successi. "
          "Va confrontato con «hash per successo» della ricerca casuale."]
    (out / "report.md").write_text("\n".join(L) + "\n")
