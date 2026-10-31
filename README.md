# Un modello di ML sceglie nonce SHA-256 migliori del caso?

Esperimento riproducibile che cerca di **falsificare** l'ipotesi: «esiste una regolarità nel
doppio SHA-256 che un modello può imparare per scegliere nonce con hash sotto soglia più spesso
della ricerca casuale o sequenziale». Dati di prova, nessun mining reale, nessuna rete.

## Avvio

```bash
cd ~/sha256-ml-esperimento
uv venv --python /opt/homebrew/bin/python3.13 .venv          # solo la prima volta
uv pip install --python .venv/bin/python numpy scikit-learn scipy matplotlib pytest
.venv/bin/python -m pytest -q                                  # test (vettori NIST inclusi)
.venv/bin/python run_experiment.py --rapido --out risultati_rapido   # prova veloce
.venv/bin/python run_experiment.py                             # completo → risultati/report.md
.venv/bin/python run_experiment.py --k 12 16 20 --seed 7       # altre soglie, altro seed
.venv/bin/python run_experiment.py --oracolo scrypt --out risultati_scrypt   # scrypt (~100 min)
.venv/bin/python run_ottimizzato.py                            # risparmio per nonce (~1 min)
.venv/bin/python run_midstate.py                               # studio 4-5 round (~12 min)
```

## Funzione obiettivo
`SHA256(SHA256(prefisso_76_byte + nonce_8_byte_little_endian))`; successo se l'hash letto
big-endian è `< 1 << (256 - k)`. Ogni candidato scelto da un modello viene **ricalcolato** con
`hashlib` prima di essere contato (`dataset.verify`).

## Disegno (pre-registrato: fissato prima di guardare i risultati)
- **Unità indipendente: il prefisso.** Train, validazione e test hanno prefissi diversi.
- **Selettore:** per ogni prefisso di test il modello ordina N candidati casuali; si verificano i
  primi M (default N=16 384, M=1 024). Confronto con M casuali dello stesso pool e con i nonce
  sequenziali 0…M-1.
- **Modelli:** classe maggioritaria, regressione logistica, gradient boosting, MLP
  (scikit-learn), ognuno con 3 seed; iperparametri fissi.
- **Ipotesi nulla:** tasso di successo = 2^-k. **Test principale:** binomiale unilaterale;
  secondario: z-test contro la ricerca casuale. α = 0,05 con Bonferroni sui confronti della corsa.
  «Significativo» richiede entrambi.
- **Potenza:** il report indica il miglioramento minimo rilevabile all'80%.
- **Controlli:** etichette mescolate (deve dare il caso); classe maggioritaria (= caso);
  oracolo «piantato», funzione finta con vantaggio ×2 noto (deve essere trovato);
  scala dei round ridotti di SHA-256 singolo e doppio (dove si perde la struttura).

## Risultati (corsa completa del 31/10/2026)
Report in `risultati/report.md`, log in `risultati/log_corsa_completa.txt`.
- **Controllo positivo superato:** sull'oracolo «piantato» i modelli trovano il vantaggio
  (logistica ×1,91 sul caso, p ≈ 6·10⁻⁴⁰; AUC al massimo teorico).
- **SHA-256 doppio reale:** nessun modello batte il caso. A k=12 sarebbe stato visto un
  miglioramento oltre il 23% con potenza 80%; a k=16 solo oltre il 99%.
- **Costo:** valutare un candidato col modello costa più che calcolarne l'hash.
- **Round ridotti:** SHA-256 singolo prevedibile a 4 round, non a 5; il doppio mai.

### Studio «midstate» (`run_midstate.py` → `risultati_midstate/`)
Con le quantità che un miner calcola prima del nonce (midstate, stato dopo i round 0-2 del
blocco 2, costante del round 3), lo SHA-256 singolo a **4 round** diventa prevedibile anche su
prefissi nuovi (AUC 0,998 gbm, 0,994 mlp). A 5 round e oltre, e per il doppio SHA-256, AUC ≤ 0,505
e nessuna cella significativa su 54. Le tre ipotesi pre-registrate sono confermate. A 4 round,
dopo il precalcolo, l'hash è una somma a 32 bit: il modello costa più dell'hash.

### Scrypt, Litecoin/Dogecoin (`run_experiment.py --oracolo scrypt` → `risultati_scrypt/`)
k=8, 2000 prefissi di test, 1,02 milioni di hash verificati per metodo: **nessun modello batte il
caso** (tutti fra 3912 e 4094 successi contro 4000 attesi). Miglioramento minimo rilevabile +6%.
Durata 104 minuti.

### Calcolare meno per nonce (`run_ottimizzato.py` → `risultati_ottimizzato/`)
L'unico vantaggio reale: midstate, round e schedule precalcolati, uscita anticipata su H7,
AsicBoost. Formato Bitcoin vero (80 byte, nonce 4 byte). 0 discordanze con hashlib su un
milione di nonce; lavoro per nonce −36,6% (−43,6% con AsicBoost a 4 header) secondo il modello
di costo. Tecniche già presenti negli ASIC: spiegano il mining, non danno un vantaggio.

## File
| file | ruolo |
|---|---|
| `config.py` | parametri e seed |
| `hashing.py` | hashlib + SHA-256 numpy a round configurabili |
| `dataset.py` | oracoli, prefissi, feature, split, verifica |
| `baselines.py` | ricerca sequenziale e casuale |
| `model.py` | modelli e ordinamento dei candidati |
| `evaluate.py` | selettore, metriche, bootstrap, test, potenza |
| `round_ridotti.py` | studio dei round ridotti |
| `ottimizzato.py`, `run_ottimizzato.py` | hash ottimizzato per nonce e misura del risparmio |
| `midstate.py`, `run_midstate.py` | studio fra 4 e 5 round con feature precalcolate |
| `report.py` | report Markdown e grafici |
| `run_experiment.py` | esegue tutto con un comando |
