import pandas as pd
import numpy as np
def calculate_net_cashflow(
    nominale,
    cedola_percentuale,
    aliquota_fiscale=0.125
):
    """
    Calcola il flusso cedolare netto annuo teorico.

    nominale: valore nominale dei titoli acquistati
    cedola_percentuale: tasso cedolare annuo in percentuale
    aliquota_fiscale: aliquota sulle cedole (0.125 = 12,5%)

    Non include ratei, commissioni, imposte sul patrimonio
    o plus/minusvalenze.
    """

    nominale = float(nominale)
    cedola_percentuale = float(cedola_percentuale)
    aliquota_fiscale = float(aliquota_fiscale)

    if nominale < 0:
        raise ValueError("Il nominale non può essere negativo")

    if cedola_percentuale < 0:
        raise ValueError("La cedola non può essere negativa")

    if not 0 <= aliquota_fiscale <= 1:
        raise ValueError("Aliquota fiscale non valida")

    cashflow_lordo = (
        nominale * cedola_percentuale / 100
    )

    cashflow_netto = (
        cashflow_lordo * (1 - aliquota_fiscale)
    )

    return cashflow_netto

def allocate_lots(
    candidates,
    amount,
    max_positions=3,
    profile="rendimento"
):
    """
    Alloca il capitale disponibile su più BTP,
    rispettando il lotto nominale minimo di €1.000.

    L'ordine dei candidati viene adattato al profilo:
    - cash_flow: privilegia il Cash Flow
    - rendimento: privilegia il rendimento
    - ladder: privilegia la diversificazione

    I candidati devono contenere i relativi score.
    """

    if candidates.empty or amount <= 0:
        return pd.DataFrame(), float(amount)

    df = candidates.copy()

    LOTTO_NOMINALE = 1000.0

    if "Prezzo" not in df.columns:
        return pd.DataFrame(), float(amount)

    df["Prezzo"] = pd.to_numeric(
        df["Prezzo"],
        errors="coerce"
    )

    df = df[
        df["Prezzo"].notna()
        & (df["Prezzo"] > 0)
    ].copy()

    if df.empty:
        return pd.DataFrame(), float(amount)

    # Ordina i candidati in funzione del profilo selezionato
    profile_score = {
        "cash_flow": "Score_CashFlow",
        "rendimento": "Score_Yield",
        "ladder": "Score_Diversificazione"
    }.get(profile, "Score_Reinvestimento")

    if profile_score in df.columns:
        df = df.sort_values(
            profile_score,
            ascending=False
        ).reset_index(drop=True)
    elif "Score_Reinvestimento" in df.columns:
        df = df.sort_values(
            "Score_Reinvestimento",
            ascending=False
        ).reset_index(drop=True)

    df["Capitale_Per_Lotto"] = (
        LOTTO_NOMINALE * df["Prezzo"] / 100
    )

    capitale_residuo = float(amount)
    allocazioni = []

    # Primo giro:
    # una posizione per ciascun candidato, fino al numero massimo.
    for _, row in df.iterrows():

        if len(allocazioni) >= max_positions:
            break

        costo_lotto = float(
            row["Capitale_Per_Lotto"]
        )

        if costo_lotto > capitale_residuo:
            continue

        allocazioni.append({
            "ISIN": row["ISIN"],
            "Descrizione": row.get("Descrizione", ""),
            "Prezzo": row["Prezzo"],
            "Lotti_Allocati": 1,
            "Nominale_Allocato": LOTTO_NOMINALE,
            "Capitale_Investito": costo_lotto
        })

        capitale_residuo -= costo_lotto

    # Secondo giro:
    # utilizza la liquidità residua seguendo l'ordine
    # determinato dal profilo.

    # Secondo giro: distribuzione dei lotti residui
    if profile == "ladder":
        # Nel profilo ladder privilegia il titolo
        # con il minore capitale già allocato.
        while True:
            acquistato = False

            ordine = sorted(
                range(len(allocazioni)),
                key=lambda i: (
                    allocazioni[i]["Capitale_Investito"],
                    i
                )
            )

            for i in ordine:
                costo_lotto = (
                    float(allocazioni[i]["Prezzo"])
                    * LOTTO_NOMINALE
                    / 100
                )

                if costo_lotto <= capitale_residuo:
                    allocazioni[i]["Lotti_Allocati"] += 1
                    allocazioni[i]["Nominale_Allocato"] += LOTTO_NOMINALE
                    allocazioni[i]["Capitale_Investito"] += costo_lotto

                    capitale_residuo -= costo_lotto
                    acquistato = True
                    break

            if not acquistato:
                break

    else:
        # Limite di concentrazione per cash_flow e rendimento.
        # Il 50% è calcolato sul capitale iniziale.
        limite_concentrazione = float(amount) * 0.50

        while True:
            acquistato = False

            for i in range(len(allocazioni)):
                costo_lotto = (
                    float(allocazioni[i]["Prezzo"])
                    * LOTTO_NOMINALE
                    / 100
                )

                nuovo_investimento = (
                    allocazioni[i]["Capitale_Investito"]
                    + costo_lotto
                )

                if (
                    costo_lotto <= capitale_residuo
                    and nuovo_investimento
                    <= limite_concentrazione
                ):
                    allocazioni[i]["Lotti_Allocati"] += 1
                    allocazioni[i]["Nominale_Allocato"] += LOTTO_NOMINALE
                    allocazioni[i]["Capitale_Investito"] += costo_lotto

                    capitale_residuo -= costo_lotto
                    acquistato = True
                    break

            if not acquistato:
                break
    # Controllo finale della concentrazione effettiva.
    # Si applica solo a cash_flow e rendimento.
    if profile in ("cash_flow", "rendimento") and len(allocazioni) >= 3:
        while True:
            totale_investito = sum(
                posizione["Capitale_Investito"]
                for posizione in allocazioni
            )

            if totale_investito <= 0:
                break

            indice_massimo = max(
                range(len(allocazioni)),
                key=lambda i: allocazioni[i]["Capitale_Investito"]
            )

            posizione = allocazioni[indice_massimo]

            peso_massimo = (
                posizione["Capitale_Investito"] / totale_investito
            )

            if peso_massimo <= 0.50 + 1e-9:
                break

            if posizione["Lotti_Allocati"] <= 1:
                break

            costo_lotto = (
                float(posizione["Prezzo"])
                * LOTTO_NOMINALE
                / 100
            )

            posizione["Lotti_Allocati"] -= 1
            posizione["Nominale_Allocato"] -= LOTTO_NOMINALE
            posizione["Capitale_Investito"] -= costo_lotto
            capitale_residuo += costo_lotto

    # Recupera la liquidità residua senza violare il limite
    # del 50% sul capitale effettivamente investito.
    if profile in ("cash_flow", "rendimento") and len(allocazioni) >= 3:
        while True:
            totale_investito = sum(
                posizione["Capitale_Investito"]
                for posizione in allocazioni
            )

            acquistato = False

            # Mantiene la priorità originale dei candidati.
            for posizione in allocazioni:
                costo_lotto = (
                    float(posizione["Prezzo"])
                    * LOTTO_NOMINALE
                    / 100
                )

                if costo_lotto > capitale_residuo:
                    continue

                nuovo_totale = totale_investito + costo_lotto
                nuovo_capitale = (
                    posizione["Capitale_Investito"] + costo_lotto
                )

                # Verifica tutte le posizioni dopo l'acquisto.
                peso_massimo = max(
                    (
                        nuovo_capitale
                        if altra is posizione
                        else altra["Capitale_Investito"]
                    ) / nuovo_totale
                    for altra in allocazioni
                )

                if peso_massimo <= 0.50 + 1e-9:
                    posizione["Lotti_Allocati"] += 1
                    posizione["Nominale_Allocato"] += LOTTO_NOMINALE
                    posizione["Capitale_Investito"] += costo_lotto
                    capitale_residuo -= costo_lotto
                    acquistato = True
                    break

            if not acquistato:
                break

    # Ottimizzazione locale: prova a scambiare un lotto
    # tra due posizioni per investire più capitale.
    if profile in ("cash_flow", "rendimento") and len(allocazioni) >= 3:
        while True:
            miglior_scambio = None
            miglior_incremento = 0.0

            totale_investito = sum(
                p["Capitale_Investito"] for p in allocazioni
            )

            for i, origine in enumerate(allocazioni):
                if origine["Lotti_Allocati"] <= 1:
                    continue

                costo_origine = (
                    float(origine["Prezzo"]) * LOTTO_NOMINALE / 100
                )

                for j, destinazione in enumerate(allocazioni):
                    if i == j:
                        continue

                    costo_destinazione = (
                        float(destinazione["Prezzo"])
                        * LOTTO_NOMINALE / 100
                    )

                    incremento = costo_destinazione - costo_origine

                    if (
                        incremento <= miglior_incremento + 1e-9
                        or incremento > capitale_residuo + 1e-9
                    ):
                        continue

                    nuovo_totale = totale_investito + incremento

                    if nuovo_totale <= 0:
                        continue

                    peso_massimo = max(
                        (
                            p["Capitale_Investito"]
                            - (costo_origine if k == i else 0)
                            + (costo_destinazione if k == j else 0)
                        ) / nuovo_totale
                        for k, p in enumerate(allocazioni)
                    )

                    if peso_massimo <= 0.50 + 1e-9:
                        miglior_scambio = (i, j, incremento)
                        miglior_incremento = incremento

            if miglior_scambio is None:
                break

            i, j, incremento = miglior_scambio

            costo_origine = (
                float(allocazioni[i]["Prezzo"])
                * LOTTO_NOMINALE / 100
            )
            costo_destinazione = (
                float(allocazioni[j]["Prezzo"])
                * LOTTO_NOMINALE / 100
            )

            allocazioni[i]["Lotti_Allocati"] -= 1
            allocazioni[i]["Nominale_Allocato"] -= LOTTO_NOMINALE
            allocazioni[i]["Capitale_Investito"] -= costo_origine

            allocazioni[j]["Lotti_Allocati"] += 1
            allocazioni[j]["Nominale_Allocato"] += LOTTO_NOMINALE
            allocazioni[j]["Capitale_Investito"] += costo_destinazione

            capitale_residuo -= incremento

    result = pd.DataFrame(allocazioni)

    if not result.empty:
        result["Liquidita_Residua"] = capitale_residuo
        result["Profilo"] = profile

    return result, capitale_residuo

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

    # Lotto operativo minimo per i Bond
    LOTTO_NOMINALE = 1000.0
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
    # Numero massimo di lotti acquistabili per ciascun Bond
    if "Prezzo" in df.columns:
        capitale_per_lotto = (
            LOTTO_NOMINALE * df["Prezzo"] / 100
        )

        df["Capitale_Per_Lotto"] = capitale_per_lotto

        df["Lotti_Acquistabili"] = np.floor(
            amount / df["Capitale_Per_Lotto"]
        ).fillna(0).astype(int)

        df["Nominale_Acquistabile"] = (
            df["Lotti_Acquistabili"]
            * LOTTO_NOMINALE
        )
    else:
        df["Lotti_Acquistabili"] = 0
        df["Nominale_Acquistabile"] = 0.0
    # Cash Flow annuo teorico generato dall'importo reinvestito
    if "Cedola" in df.columns:
        df["Cedola"] = pd.to_numeric(
            df["Cedola"],
            errors="coerce"
        )

        df["CashFlow_Annuale"] = (
            df["Nominale_Acquistabile"]
            * df["Cedola"]
            / 100
        )
    else:
        df["CashFlow_Annuale"] = 0.0
    # Cash flow netto annuo teorico dei BTP italiani
    df["CashFlow_Netto_Annuale"] = df.apply(
        lambda row: calculate_net_cashflow(
            nominale=row["Nominale_Acquistabile"],
            cedola_percentuale=(
                row["Cedola"]
                if pd.notna(row["Cedola"])
                else 0.0
            ),
            aliquota_fiscale=0.125
        ),
        axis=1
)
    # Capitale effettivamente investito
    if "Prezzo" in df.columns:
        df["Capitale_Investito"] = (
            df["Nominale_Acquistabile"]
            * df["Prezzo"]
            / 100
        )
    else:
        df["Capitale_Investito"] = 0.0

    # Liquidità residua dopo il reinvestimento
    df["Liquidita_Residua"] = (
        amount - df["Capitale_Investito"]
    )

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
            "Nominale_Acquistabile",
            "Capitale_Per_Lotto",
            "Lotti_Acquistabili",
            "Capitale_Investito",
            "Liquidita_Residua",
            "Cedola",
            "Yield",
            "Duration",
            "Scadenza",
            "Rating",
            "Cedola_Reinvestita",
            "CashFlow_Annuale",
            "CashFlow_Netto_Annuale",
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
def build_reinvestment_allocation(
    catalog,
    amount,
    bonds=None,
    profile="rendimento",
    max_positions=3
):
    """
    Costruisce una simulazione completa di reinvestimento:

    1. calcola i candidati in funzione del profilo;
    2. ordina i candidati;
    3. distribuisce il capitale su più BTP;
    4. restituisce allocazione e liquidità residua.
    """

    candidates = reinvestment_candidates(
        catalog,
        amount,
        bonds=bonds,
        profile=profile
    )

    if candidates.empty:
        return pd.DataFrame(), float(amount)

    allocation, residual = allocate_lots(
        candidates,
        amount,
        max_positions=max_positions,
        profile=profile
    )

    return allocation, residual