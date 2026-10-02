from pathlib import Path

import pandas as pd


def read_bank_portfolio_csv(
    file_path,
    reference_date=None,
    holder=None,
    account=None
):
    """
    Legge il file I_miei_titoli.csv esportato dalla banca.

    Questa funzione:
    - legge il CSV bancario;
    - normalizza i nomi delle colonne;
    - converte i numeri italiani in valori numerici;
    - aggiunge data di riferimento e importazione;
    - associa titolare e conto;
    - NON modifica i dati della piattaforma.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File non trovato: {file_path}"
        )

    # Il file bancario usa separatore ;
    # e codifica UTF-8.
    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="utf-8"
    )

    # Rimuove eventuali spazi dai nomi delle colonne.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Mapping tra colonne banca e nomi interni
    # della piattaforma.
    column_mapping = {
        "Descrizione Titolo": "Descrizione",
        "ISIN": "ISIN",
        "Q.tà": "Quantita",
        "Divisa": "Valuta",
        "Prezzo carico": "Prezzo_Carico",
        "Prezzo Mercato": "Prezzo_Attuale",
        "CTV in Euro": "Valore_Attuale",
        "Var %": "Plusvalenza_Percentuale",
        "Utile/Perdita": "Plusvalenza"
    }

    missing_columns = [
        column
        for column in column_mapping
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonne mancanti nel file bancario: "
            + ", ".join(missing_columns)
        )

    df = df.rename(
        columns=column_mapping
    )

    df = df[
        list(column_mapping.values())
    ].copy()

    # Conversione numeri dal formato italiano.
    #
    # 21.000    -> 21000
    # 106,7214  -> 106.7214
    # 20.396,18 -> 20396.18

    numeric_columns = [
        "Quantita",
        "Prezzo_Carico",
        "Prezzo_Attuale",
        "Valore_Attuale",
        "Plusvalenza_Percentuale",
        "Plusvalenza"
    ]

    for column in numeric_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Normalizzazione campi testuali.

    df["ISIN"] = (
        df["ISIN"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Descrizione"] = (
        df["Descrizione"]
        .astype(str)
        .str.strip()
    )

    df["Valuta"] = (
        df["Valuta"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Data a cui si riferisce il portafoglio bancario.

    if reference_date is not None:
        reference_date = pd.to_datetime(
            reference_date,
            errors="raise"
        ).date()

        df["Data_Riferimento"] = reference_date
    else:
        df["Data_Riferimento"] = pd.NaT

    # Data e ora in cui avviene l'importazione.

    df["Data_Importazione"] = pd.Timestamp.now()

    # Associazione a titolare e conto.

    if holder is not None:
        df["Titolare"] = str(holder).strip().upper()
    else:
        df["Titolare"] = None

    if account is not None:
        df["Conto"] = str(account).strip()
    else:
        df["Conto"] = None

    return df


def classify_bank_assets(
    df,
    bond_catalog=None,
    certificates=None,
    instruments=None
):
    """
    Classifica gli strumenti importati dalla banca.

    Priorità:
    1. ISIN presente nel catalogo bond -> BOND
    2. ISIN presente nel portafoglio certificates -> CERTIFICATE
    3. Regola conservativa per i BTP
    4. Tutto il resto -> DA_CLASSIFICARE

    La funzione non modifica i file della piattaforma.
    """

    result = df.copy()

    bond_isins = set()
    certificate_isins = set()
    instrument_types = {}

    if (
        instruments is not None
        and not instruments.empty
        and "ISIN" in instruments.columns
        and "Tipo_Asset" in instruments.columns
    ):
        instrument_types = {
            str(row["ISIN"]).strip().upper():
            str(row["Tipo_Asset"]).strip().upper()
            for _, row in instruments.iterrows()
        }

    if (
        bond_catalog is not None
        and not bond_catalog.empty
        and "ISIN" in bond_catalog.columns
    ):
        bond_isins = set(
            bond_catalog["ISIN"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if (
        certificates is not None
        and not certificates.empty
        and "ISIN" in certificates.columns
    ):
        certificate_isins = set(
            certificates["ISIN"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )

    def classify_row(row):
        isin = str(
            row["ISIN"]
        ).strip().upper()

        description = str(
            row["Descrizione"]
        ).strip().upper()

        if isin in instrument_types:
            return instrument_types[isin]

        if isin in bond_isins:
            return "BOND"

        if isin in certificate_isins:
            return "CERTIFICATE"

        if description.startswith("BTP"):
            return "BOND"

        return "DA_CLASSIFICARE"

    result["Tipo_Asset"] = result.apply(
        classify_row,
        axis=1
    )

    return result