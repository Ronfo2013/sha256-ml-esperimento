# Calcolare meno per nonce: quanto si risparmia davvero

Formato Bitcoin vero (header 80 byte, nonce 4 byte LE, confronto little-endian). Ogni versione è verificata contro `hashlib` prima di essere misurata.

## Correttezza

- 1,000,000 nonce consecutivi, k=8: **0 discordanze** fra versione ottimizzata e hashlib (3857 successi, attesi 3906).
- AsicBoost con 4 header che condividono il blocco 2: **0 discordanze** su 200.000 nonce.
- Test unitari: blocco genesi di Bitcoin, k = 1, 4, 8, 12, AsicBoost (`tests/test_ottimizzato.py`).

## Lavoro per nonce (modello di costo software)

Round = 38 operazioni a 32 bit, parola di schedule = 21, somma di feed-forward = 1.

| versione | round | parole di schedule | feed-forward | operazioni | risparmio |
|---|---|---|---|---|---|
| ingenuo (3 compressioni) | 192 | 144.0 | 24 | 10,344 | 0.0% |
| ottimizzato (midstate, precalcolo, uscita anticipata) | 122 | 91.0 | 9 | 6,556 | 36.6% |
| ottimizzato + AsicBoost (m=4) | 122 | 56.5 | 9 | 5,832 | 43.6% |

## Tempi misurati (un core, secondi per nonce)

| implementazione | µs per nonce | rispetto a numpy ingenuo |
|---|---|---|
| numpy ingenuo | 3.279 | 1,00 |
| numpy ottimizzato | 1.652 | 0.50 |
| numpy ottimizzato + AsicBoost (m=4) | 1.437 | 0.44 |
| hashlib (C, ciclo Python, riferimento) | 0.786 | — |

`prepare` (midstate e precalcolo, una volta per header): 1431 µs.

## Lettura

- Il risparmio viene da tre fatti strutturali: il primo blocco non contiene il nonce, molte parole del blocco 2 e del secondo hash sono padding costante, e Bitcoin guarda solo l'ultima parola (H7), che è pronta tre round prima della fine.
- Il midstate da solo vale circa un terzo. Il resto aggiunge pochi punti percentuali.
- AsicBoost condivide solo lo schedule del blocco 2 e richiede di trovare radici di Merkle con gli stessi ultimi 4 byte (lavoro extra non contato qui). In hardware il guadagno dichiarato è ~20% perché lì lo schedule pesa di più che in questo modello.
- Tutto questo è già nei miner ASIC e nei software di mining da anni: il risultato spiega come funzionano, non dà un vantaggio su di loro.
- I tempi numpy includono overhead di memoria e Python; contano i rapporti, non i valori assoluti. hashlib è C ottimizzato chiamato nonce per nonce.
- Il rapporto misurato (circa 0,5) è migliore di quello previsto dal modello (0,63) perché la versione numpy ingenua costruisce anche messaggio e padding a ogni nonce: il numero da citare è quello del modello di costo, non il tempo.
