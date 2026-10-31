"""Esegue lo studio midstate (`midstate.study`) e scrive midstate.json, PNG e report.md."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

import midstate

SOGLIA_H1 = 0.55   # "nettamente > 0,5": AUC almeno 0,55 oltre che significativa (operativizzazione)
SOGLIA_NULLA = 0.52  # "~0,5": nessuna cella sopra 0,52


def annota(rows: list[dict]) -> None:
    """p unilaterale AUC > 0,5 da z = (auc - 0,5) / se_bootstrap, con Bonferroni sulle celle."""
    for r in rows:
        r["z"] = (r["auc"] - 0.5) / r["auc_se"] if r["auc_se"] > 0 else float("nan")
        r["p"] = float(norm.sf(r["z"]))
        r["p_bonf"] = min(1.0, r["p"] * len(rows))


def verdetti(rows: list[dict]) -> list[tuple[str, str, str]]:
    def sig(r):
        return r["p_bonf"] < 0.05

    h1 = [r for r in rows if r["funzione"] == "singolo" and r["round"] == 4 and r["feature"] == "precalcolo"]
    h2 = [r for r in rows if r["funzione"] == "singolo" and r["round"] == 4 and r["feature"] == "nonce"]
    h3 = [r for r in rows if not (r["funzione"] == "singolo" and r["round"] == 4)]
    out = []
    best = max(h1, key=lambda r: r["auc"])
    ok = any(sig(r) and r["auc"] >= SOGLIA_H1 for r in h1)
    out.append(("H1", "confermata" if ok else "smentita",
                f"migliore: {best['modello']} AUC {best['auc']:.3f}, p Bonferroni {best['p_bonf']:.2g} "
                f"(criterio: p_bonf < 0,05 e AUC >= {SOGLIA_H1})"))
    best = max(h2, key=lambda r: r["auc"])
    ok = not any(sig(r) for r in h2) and best["auc"] <= SOGLIA_NULLA
    out.append(("H2", "confermata" if ok else "smentita",
                f"massima AUC {best['auc']:.3f} ({best['modello']}), p Bonferroni {best['p_bonf']:.2g} "
                f"(criterio: nessuna cella significativa e AUC <= {SOGLIA_NULLA})"))
    best = max(h3, key=lambda r: r["auc"])
    ko = [r for r in h3 if sig(r)]
    ok = not ko and best["auc"] <= SOGLIA_NULLA
    out.append(("H3", "confermata" if ok else "smentita",
                f"{len(h3)} celle, massima AUC {best['auc']:.3f} ({best['funzione']} R={best['round']} "
                f"{best['feature']} {best['modello']}), celle significative: {len(ko)} "
                f"(criterio: nessuna significativa e AUC <= {SOGLIA_NULLA})"))
    return out


def grafico(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, len(midstate.MODELS), figsize=(13, 7), sharey=True)
    for i, fun in enumerate(("singolo", "doppio")):
        for j, mod in enumerate(midstate.MODELS):
            ax = axes[i][j]
            for k, fs in enumerate(midstate.FEATURE_SETS):
                rr = sorted((r for r in rows if r["funzione"] == fun and r["modello"] == mod
                             and r["feature"] == fs), key=lambda r: r["round"])
                if not rr:
                    continue
                x = [r["round"] + (k - 1) * 0.08 for r in rr]
                y = [r["auc"] for r in rr]
                err = [[r["auc"] - r["ic_basso"] for r in rr], [r["ic_alto"] - r["auc"] for r in rr]]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=fs)
            ax.axhline(0.5, color="grey", ls="--", lw=1)
            ax.set_title(f"{fun} - {mod}")
            ax.set_xlabel("round")
            if j == 0:
                ax.set_ylabel("AUC (IC 95%)")
    axes[0][0].legend(title="feature")
    fig.suptitle("Midstate: AUC su prefissi nuovi, soglia k=4")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def ipotesi_da_docstring() -> str:
    m = re.search(r"Ipotesi pre-registrate.*?(?=\nRegola dell)", midstate.__doc__, re.S)
    return m.group(0).strip()


def report(rows: list[dict], par: dict, secondi: float, path: Path) -> None:
    L = ["# Studio midstate: fra 4 e 5 round", "",
         f"Corsa del {time.strftime('%d/%m/%Y')}, {secondi / 60:.1f} minuti. Parametri: {par}.", ""]
    if par["n_train"] != 3000 or par["n_test"] != 1000:
        L += [f"**Attenzione**: parametri ridotti rispetto al default (3000 prefissi di train, "
              f"1000 di test): {par['motivo']}.", ""]
    L += ["## Ipotesi pre-registrate (copiate da `midstate.py`)", "", "```", ipotesi_da_docstring(),
          "```", "", "## Verdetti", ""]
    for h, v, d in verdetti(rows):
        L.append(f"- **{h}: {v}.** {d}")
    L += ["", "I criteri numerici (AUC >= 0,55 per H1, <= 0,52 per il \"~0,5\" di H2 e H3) sono "
          "l'operativizzazione di «nettamente» e «~», fissata nello script prima di vedere i numeri.", "",
          "## Tabella completa", "",
          f"p unilaterale AUC > 0,5 da z = (AUC - 0,5) / se_bootstrap; Bonferroni su {len(rows)} celle.", "",
          "| funzione | round | feature | modello | AUC | IC 95% | se | z | p | p Bonf. | lift top 1/16 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['funzione']} | {r['round']} | {r['feature']} | {r['modello']} | {r['auc']:.4f} | "
                 f"{r['ic_basso']:.3f}-{r['ic_alto']:.3f} | {r['auc_se']:.4f} | {r['z']:.2f} | "
                 f"{r['p']:.3g} | {r['p_bonf']:.3g} | {r['lift_top16']:.2f} |")
    L += ["", "![AUC per round](midstate.png)", "",
          "## Note", "",
          "- **Costo.** A 4 round, dopo il precalcolo, l'hash è una somma a 32 bit "
          "(`H0 = D_a + W3`, verificata nei test). Qualunque modello costa più di calcolarlo: "
          "anche un modello che la imparasse non darebbe alcun vantaggio pratico. Il modello "
          "inoltre non calcola mai l'hash: riceve solo quantità indipendenti dal nonce.",
          "- **Metodo.** Le AUC sono a k=4 (soglia facile, successo con p=1/16) per avere molti "
          "positivi; non sono la soglia dell'esperimento principale. Regime: prefissi nuovi "
          "(train e test non condividono prefissi), IC e se da bootstrap a grappolo sui prefissi. "
          "Iperparametri fissi, non toccati dopo aver visto il test.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rapido", action="store_true", help="pochi prefissi, prova in 1-2 minuti")
    ap.add_argument("--out", default="risultati_midstate")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--motivo", default="", help="perché i parametri sono ridotti (va nel report)")
    a = ap.parse_args()
    n_train, n_test, reps = (60, 30, 100) if a.rapido else (a.n_train, a.n_test, 500)
    motivo = "prova rapida" if a.rapido else a.motivo
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    rows = midstate.study(n_train=n_train, n_test=n_test, reps=reps)
    secondi = time.perf_counter() - t
    annota(rows)
    par = dict(n_train=n_train, n_test=n_test, per_prefix=64, reps=reps, motivo=motivo)
    (out / "midstate.json").write_text(json.dumps(dict(parametri=par, secondi=secondi, righe=rows),
                                                  indent=1), encoding="utf-8")
    grafico(rows, out / "midstate.png")
    report(rows, par, secondi, out / "report.md")
    print(f"fatto in {secondi:.0f} s -> {out}/report.md")


if __name__ == "__main__":
    main()
