# V1 App

## Avvio locale

Dalla root del repository:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app/app.py
```

Il browser aprirà la dashboard locale.

## Deploy con Streamlit Community Cloud

1. Pubblica la repository su GitHub.
2. Vai su Streamlit Community Cloud.
3. Crea una nuova app indicando repository, branch e file:
   `app/app.py`
4. Le dipendenze vengono installate da `requirements.txt`.

I CSV sono letti dalla cartella `data/`. La V1 non modifica automaticamente i CSV.
