import pandas as pd


def reinvestment_candidates(
    catalog,
    amount,
    bonds=None,
    profile="rendimento"
):
    """
     Genera una lista di possibili reinvestimenti delle cedole.

    bonds contiene il portafoglio Bond attualmente posseduto
    e verrà utilizzato per valutare la concentrazione.
    """

    if catalog.empty:
        return pd.DataFrame()

    df = catalog.copy()

    # Profili di reinvestimento
    profiles = {
        "cash_flow": {
            "yield": 0.20,
            "cash_flow": 0.40,
            "diversificazione": 0.15,
            "rating": 0.15,
            "prezzo": 0.10
        },
        "rendimento": {
            "yield": 0.40,
            "cash_flow": 0.20,
            "diversificazione": 0.20,
            "rating": 0.10,
            "prezzo": 0.10
        },
        "ladder": {
            "yield": 0.25,
            "cash_flow": 0.15,
            "diversificazione": 0.40,
            "rating": 0.10,
            "prezzo": 0.10
        }
    }

    # Se il profilo non è riconosciuto,
    # utilizza il profilo rendimento come default.
    if profile not in profiles:
        profile = "rendimento"

    weights = profiles[profile]

    # Analisi della concentrazione temporale del portafoglio
    maturity_concentration = {}

    if bonds is not None and not bonds.empty:
        portfolio = bonds.copy()

        if "Data_Scadenza" in portfolio.columns:
            portfolio["Data_Scadenza"] = pd.to_datetime(
                portfolio["Data_Scadenza"],
                errors="coerce"
            )

            today = pd.Timestamp.today().normalize()

            portfolio["Giorni_Scadenza_Reinvestimento"] = (
                portfolio["Data_Scadenza"] - today
            ).dt.days

            portfolio["Fascia_Reinvestimento"] = pd.cut(
                portfolio["Giorni_Scadenza_Reinvestimento"],
                bins=[-1, 730, 1825, 3650, float("inf")],
                labels=[
                    "< 2 anni",
                    "2–5 anni",
                    "5–10 anni",
                    "> 10 anni"
                ]
            )

            if "Valore_Attuale_Calc" in portfolio.columns:
                maturity_concentration = (
                    portfolio
                    .groupby(
                        "Fascia_Reinvestimento",
                        observed=False
                    )["Valore_Attuale_Calc"]
                    .sum()
                    .to_dict()
                )
    # Calcola la percentuale di concentrazione per fascia
    total_bond_value = sum(maturity_concentration.values())

    maturity_weights = {}

    if total_bond_value > 0:
        maturity_weights = {
            fascia: valore / total_bond_value
            for fascia, valore in maturity_concentration.items()
        }
    # Valuta quanto una nuova scadenza aiuta a diversificare
    def maturity_diversification_score(maturity):
        if pd.isna(maturity):
            return 0.0

        days = (
            pd.to_datetime(maturity, errors="coerce")
            - pd.Timestamp.today().normalize()
        ).days

        if days < 0:
            return 0.0

        if days <= 730:
            fascia = "< 2 anni"
        elif days <= 1825:
            fascia = "2–5 anni"
        elif days <= 3650:
            fascia = "5–10 anni"
        else:
            fascia = "> 10 anni"

        current_weight = maturity_weights.get(fascia, 0.0)

        # Più una fascia è poco rappresentata,
        # maggiore è il beneficio della diversificazione.
        return (1.0 - current_weight) * 100
    # Score di diversificazione temporale per ogni Bond candidato
    if "Scadenza" in df.columns:
        df["Score_Diversificazione"] = (
            df["Scadenza"].apply(
                maturity_diversification_score
            )
        )
    else:
        df["Score_Diversificazione"] = 0.0
    # Normalizzazione numerica
    df["Yield"] = pd.to_numeric(
        df["Yield"],
        errors="coerce"
    )

    df["Duration"] = pd.to_numeric(
        df["Duration"],
        errors="coerce"
    )
    # Score rendimento normalizzato 0-100
    if df["Yield"].notna().any():
        min_yield = df["Yield"].min()
        max_yield = df["Yield"].max()

        if max_yield > min_yield:
            df["Score_Yield"] = (
                (df["Yield"] - min_yield)
                / (max_yield - min_yield)
                * 100
            )
        else:
            df["Score_Yield"] = 100.0
    else:
        df["Score_Yield"] = 0.0

    # Cash Flow annuo teorico generato dall'importo reinvestito
    if "Cedola" in df.columns:
        df["Cedola"] = pd.to_numeric(
            df["Cedola"],
            errors="coerce"
        )

        df["CashFlow_Annuale"] = (
            amount * df["Cedola"] / 100
        )
    else:
        df["CashFlow_Annuale"] = 0.0
    # Score Prezzo 0-100
    if "Prezzo" in df.columns:
        df["Prezzo"] = pd.to_numeric(
            df["Prezzo"],
            errors="coerce"
        )

        df["Score_Prezzo"] = (
            (100 - df["Prezzo"])
            .clip(lower=0)
            .clip(upper=100)
        )
    else:
        df["Score_Prezzo"] = 0.0

  # Score Rating normalizzato 0-100
    rating_order = {
        "AAA": 100,
        "AA+": 95,
        "AA": 90,
        "AA-": 85,
        "A+": 80,
        "A": 75,
        "A-": 70,
        "BBB+": 65,
        "BBB": 60,
        "BBB-": 55,
        "BB+": 45,
        "BB": 40,
        "BB-": 35,
        "B+": 25,
        "B": 20,
        "B-": 15
    }

    if "Rating" in df.columns:
        df["Score_Rating"] = (
            df["Rating"]
            .astype(str)
            .str.strip()
            .map(rating_order)
            .fillna(0)
        )
    else:
        df["Score_Rating"] = 0.0

    # Score Cash Flow normalizzato 0-100
    if df["CashFlow_Annuale"].notna().any():
        min_cashflow = df["CashFlow_Annuale"].min()
        max_cashflow = df["CashFlow_Annuale"].max()

        if max_cashflow > min_cashflow:
            df["Score_CashFlow"] = (
                (df["CashFlow_Annuale"] - min_cashflow)
                / (max_cashflow - min_cashflow)
                * 100
            )
        else:
            df["Score_CashFlow"] = 100.0
    else:
        df["Score_CashFlow"] = 0.0
    # Contributi dei singoli fattori allo score finale
    df["Contributo_Yield"] = (
        df["Score_Yield"] * weights["yield"]
    )

    df["Contributo_CashFlow"] = (
        df["Score_CashFlow"] * weights["cash_flow"]
    )

    df["Contributo_Diversificazione"] = (
        df["Score_Diversificazione"]
        * weights["diversificazione"]
    )

    df["Contributo_Rating"] = (
        df["Score_Rating"] * weights["rating"]
    )

    df["Contributo_Prezzo"] = (
        df["Score_Prezzo"] * weights["prezzo"]
    )

    # Score Reinvestimento complessivo
    df["Score_Reinvestimento"] = (
        df["Contributo_Yield"]
        + df["Contributo_CashFlow"]
        + df["Contributo_Diversificazione"]
        + df["Contributo_Rating"]
        + df["Contributo_Prezzo"]
    )

    # Motivazione dettagliata della raccomandazione
    def build_recommendation_reason(row):
        contributions = {
            "Yield": row["Contributo_Yield"],
            "Cash Flow": row["Contributo_CashFlow"],
            "Diversificazione": row["Contributo_Diversificazione"],
            "Rating": row["Contributo_Rating"],
            "Prezzo": row["Contributo_Prezzo"]
        }

        main_factor = max(
            contributions,
            key=contributions.get
        )

        reason = (
            f"Score {row['Score_Reinvestimento']:.2f}: "
            f"Yield +{row['Contributo_Yield']:.2f}; "
            f"Cash Flow +{row['Contributo_CashFlow']:.2f}; "
            f"Diversificazione +{row['Contributo_Diversificazione']:.2f}; "
            f"Rating +{row['Contributo_Rating']:.2f}; "
            f"Prezzo +{row['Contributo_Prezzo']:.2f}. "
        )

        reason += (
            f"Driver principale: {main_factor}."
        )

        return reason

    df["Motivazione"] = df.apply(
        build_recommendation_reason,
        axis=1
    )

    df = df.sort_values(
        "Score_Reinvestimento",
        ascending=False
    )

    df["Cedola_Reinvestita"] = amount

    return df[
        [
            "ISIN",
            "Descrizione",
            "Prezzo",
            "Cedola",
            "Yield",
            "Duration",
            "Scadenza",
            "Rating",
            "Cedola_Reinvestita",
            "CashFlow_Annuale",
            "Score_Yield",
            "Score_Diversificazione",
            "Score_Rating",
            "Score_Prezzo",
            "Score_CashFlow",
            "Score_Reinvestimento",
            "Contributo_Yield",
            "Contributo_CashFlow",
            "Contributo_Diversificazione",
            "Contributo_Rating",
            "Contributo_Prezzo",
            "Motivazione"
        ]
    ].head(5)    