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


def scrypt(cfg: Config, quick: bool = False) -> Config:
    """Dimensioni per scrypt (~260 µs/hash su 1 core contro ~0,6 µs del doppio SHA-256), fissate
    PRIMA di vedere il test. k=8 (1 su 256) con k_train=4; test: 2000 prefissi x 8192 candidati,
    verificati i primi 512 (M/N = 1/16 come per SHA-256), cioè 1,02M hash verificati per metodo.
    Misurato: ~21k hash/s su 8 core; ~49M hash in tutto, circa 40 min di solo hashing."""
    cfg = cfg.with_(oracolo="scrypt", k=8, k_train=4, n_train_prefixes=250, n_val_prefixes=60,
                    nonces_per_train_prefix=4096, n_test_prefixes=2000, n_candidates=8192,
                    top_m=512)
    if quick:
        cfg = rapido(cfg).with_(k=6, k_train=3, n_train_prefixes=40, n_val_prefixes=15,
                                nonces_per_train_prefix=1024, n_test_prefixes=60,
                                n_candidates=4096, top_m=256)
    return cfg
