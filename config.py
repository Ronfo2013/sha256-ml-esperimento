"""Parametri dell'esperimento. Ogni seed parte da `master_seed`: un solo numero lo riproduce."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class Config:
    master_seed: int = 20261031
    oracolo: str = "sha256d"          # "sha256d" (reale) | "piantato" (controllo positivo)
    k: int = 12                       # successo: hash < 2^(256-k)
    k_train: int = 6                  # soglia più facile per l'addestramento (etichette più dense)
    prefix_len: int = 76

    # dataset: unità indipendente = prefisso; gli split non condividono prefissi
    n_train_prefixes: int = 400
    n_val_prefixes: int = 100
    nonces_per_train_prefix: int = 4096
    max_train_rows: int = 300_000     # sottocampiono i negativi (tengo tutti i positivi)
    use_prefix_features: bool = True  # bit della coda del prefisso (stesso blocco del nonce)

    # test: per ogni prefisso nuovo, N candidati; il selettore verifica i primi M
    n_test_prefixes: int = 1000
    n_candidates: int = 16384         # N
    top_m: int = 1024                 # M  (M/N = 1/16)

    model_seeds: tuple[int, ...] = (0, 1, 2)
    models: tuple[str, ...] = ("maggioritaria", "logistica", "gbm", "mlp")
    bootstrap_reps: int = 2000
    alpha: float = 0.05
    learning_curve_fracs: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)
    workers: int | None = None        # processi per l'hashing (None = tutti i core)

    # studio dei round ridotti (controllo positivo strutturale)
    round_list: tuple[int, ...] = (4, 5, 6, 8, 16, 64)
    round_prefixes: int = 3
    round_nonces: int = 60_000

    extra: dict = field(default_factory=dict)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)


def rapido(cfg: Config) -> Config:
    """Versione ridotta per prove veloci (minuti -> secondi)."""
    return cfg.with_(n_train_prefixes=120, n_val_prefixes=40, nonces_per_train_prefix=2048,
                     n_test_prefixes=150, n_candidates=8192, top_m=512, model_seeds=(0,),
                     bootstrap_reps=300, round_list=(4, 5, 64), round_prefixes=2,
                     round_nonces=20_000)
