# Piattaforma Investimenti

Prima versione eseguibile della piattaforma per gestione Bond e Certificates.

## Architettura

- `data/` = CSV sorgente
- `docs/` = specifiche funzionali e formule
- `rules/` = regole di scoring e rotazione
- `app/` = applicazione Streamlit
- `database/` = snapshot SQLite generato dall'app

## Avvio su Windows

Doppio clic su `run_app.bat`, oppure da PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Avvio con GitHub Codespaces

Aprire la repository in Codespaces, quindi nel terminale:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Deploy online

La repository è predisposta per Streamlit Community Cloud. Entry point:

`app/app.py`

Dependency file:

`requirements.txt`

I CSV nella cartella `data/` vengono letti automaticamente.

## Stato V1

Implementati:
- import CSV
- calcolo valore carico/attuale
- plus/minus
- yield on cost
- giorni a scadenza
- scoring bond
- candidati rotazione
- analisi worst-of e distanza barriera certificates quando i dati sono presenti
- dashboard
- calendario cedole
- snapshot SQLite

Da completare nelle iterazioni successive:
- recupero automatico prezzi
- catalogo bond con dati di mercato
- cedole realmente incassate
- motore di confronto bond candidato vs posseduto
- generazione completa delle proposte di rotazione
- storico/equity curve
- autenticazione e gestione multiutente
