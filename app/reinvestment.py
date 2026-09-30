import pandas as pd


def reinvestment_candidates(catalog, amount):
    """
    Genera una lista di possibili reinvestimenti delle cedole.

    La prima versione privilegia il rendimento,
    mantenendo i dati necessari per evoluzioni successive:
    - duration
    - scadenza
    - rating
    """

    if catalog.empty:
        return pd.DataFrame()

    df = catalog.copy()

    # Normalizzazione numerica
    df["Yield"] = pd.to_numeric(
        df["Yield"],
        errors="coerce"
    )

    df["Duration"] = pd.to_numeric(
        df["Duration"],
        errors="coerce"
    )

    # Primo modello: priorità al rendimento
    df["Score_CashFlow"] = (
        df["Yield"] * 10
    )

    df = df.sort_values(
        "Score_CashFlow",
        ascending=False
    )

    df["Cedola_Reinvestita"] = amount

    return df[
        [
            "ISIN",
            "Descrizione",
            "Yield",
            "Duration",
            "Scadenza",
            "Cedola_Reinvestita",
            "Score_CashFlow"
        ]
    ].head(5)
