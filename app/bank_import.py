from pathlib import Path
import shutil

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

    df["Data_Importazione"] = (
    pd.Timestamp.now(tz="Europe/Rome")
    .tz_localize(None)
)

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


def preview_bond_update(
    bank_df,
    current_bonds
):
    """
    Confronta i bond importati dalla banca con il
    portafoglio bond attuale.

    La funzione produce solo una preview.
    Non modifica alcun file.
    """

    bank_bonds = bank_df[
        bank_df["Tipo_Asset"] == "BOND"
    ].copy()

    current = current_bonds.copy()


    current_keys = set(
        zip(
            current["Titolare"].astype(str).str.strip().str.upper(),
            current["Conto"].astype(str).str.strip().str.upper(),
            current["ISIN"].astype(str).str.strip().str.upper()
        )
    )

    bank_keys = set(
        zip(
            bank_bonds["Titolare"].astype(str).str.strip().str.upper(),
            bank_bonds["Conto"].astype(str).str.strip().str.upper(),
            bank_bonds["ISIN"].astype(str).str.strip().str.upper()
        )
    )

    comparison_columns = {
        "Quantita": "Quantita_Nominale",
        "Prezzo_Carico": "Prezzo_Carico",
        "Prezzo_Attuale": "Prezzo_Attuale",
        "Valore_Attuale": "Valore_Attuale",
        "Plusvalenza": "Plusvalenza",
        "Plusvalenza_Percentuale": "Plusvalenza_Percentuale"
    }

    current_lookup = {}

    for _, current_row in current.iterrows():
        key = (
            str(current_row["Titolare"]).strip().upper(),
            str(current_row["Conto"]).strip().upper(),
            str(current_row["ISIN"]).strip().upper()
        )

        current_lookup[key] = current_row
    def get_changes(row):
        key = (
            str(row["Titolare"]).strip().upper(),
            str(row["Conto"]).strip().upper(),
            str(row["ISIN"]).strip().upper()
        )

        if key not in current_lookup:
            return ""

        current_row = current_lookup[key]

        changes = []

        for bank_column, current_column in comparison_columns.items():
            new_value = row[bank_column]
            old_value = current_row[current_column]

            if pd.isna(new_value) and pd.isna(old_value):
                continue

            if pd.isna(new_value) or pd.isna(old_value):
                changes.append(
                    f"{bank_column}: {old_value} -> {new_value}"
                )
                continue

            if abs(
                float(new_value) - float(old_value)
            ) > 0.01:
                changes.append(
                    f"{bank_column}: {old_value} -> {new_value}"
                )

        return " | ".join(changes)
    def determine_status(row):
        key = (
            str(row["Titolare"]).strip().upper(),
            str(row["Conto"]).strip().upper(),
            str(row["ISIN"]).strip().upper()
        )

        if key not in current_keys:
            return "NUOVO"

        current_row = current_lookup[key]

        for bank_column, current_column in comparison_columns.items():
            bank_value = row[bank_column]
            current_value = current_row[current_column]

            if pd.isna(bank_value) and pd.isna(current_value):
                continue

            if pd.isna(bank_value) or pd.isna(current_value):
                return "AGGIORNATO"

            if abs(
                float(bank_value) - float(current_value)
            ) > 0.01:
                return "AGGIORNATO"

        return "INVARIATO"

    bank_bonds["Stato"] = bank_bonds.apply(
        determine_status,
        axis=1
    )

    bank_bonds["Variazioni"] = bank_bonds.apply(
        get_changes,
        axis=1
    )

    import_scopes = set(
        zip(
            bank_bonds["Titolare"].astype(str).str.strip().str.upper(),
            bank_bonds["Conto"].astype(str).str.strip().str.upper()
        )
    )

    missing_rows = []

    for _, current_row in current.iterrows():
        scope = (
            str(current_row["Titolare"]).strip().upper(),
            str(current_row["Conto"]).strip().upper()
        )

        key = (
            str(current_row["Titolare"]).strip().upper(),
            str(current_row["Conto"]).strip().upper(),
            str(current_row["ISIN"]).strip().upper()
        )

        if scope not in import_scopes:
            continue

        if key in bank_keys:
            continue

        missing_rows.append(
            {
                "Titolare": current_row["Titolare"],
                "Conto": current_row["Conto"],
                "ISIN": current_row["ISIN"],
                "Descrizione": current_row["Descrizione"],
                "Stato": "ASSENTE_DAL_NUOVO_ESTRATTO",
                "Variazioni": ""
            }
        )
    if missing_rows:
        missing_df = pd.DataFrame(
            missing_rows
        )

        bank_bonds = pd.concat(
            [
                bank_bonds,
                missing_df
            ],
            ignore_index=True,
            sort=False
        )

    return bank_bonds


def preview_certificate_update(
    bank_df,
    current_certificates
):
    """
    Confronta i certificate importati dalla banca con il
    portafoglio certificate attuale.

    La funzione produce solo una preview.
    Non modifica alcun file.
    """

    bank_certificates = bank_df[
        bank_df["Tipo_Asset"] == "CERTIFICATE"
    ].copy()

    current = current_certificates.copy()
    bank_keys = set(
        zip(
            bank_certificates["Titolare"].astype(str).str.strip().str.upper(),
            bank_certificates["Conto"].astype(str).str.strip().str.upper(),
            bank_certificates["ISIN"].astype(str).str.strip().str.upper()
        )
    )
    comparison_columns = [
        "Quantita",
        "Prezzo_Carico",
        "Prezzo_Attuale"
    ]

    tolerance = 0.01

    def find_current_row(row):
        mask = (
            current["Titolare"]
            .astype(str)
            .str.strip()
            .str.upper()
            == str(row["Titolare"]).strip().upper()
        )

        mask &= (
            current["Conto"]
            .astype(str)
            .str.strip()
            .str.upper()
            == str(row["Conto"]).strip().upper()
        )

        mask &= (
            current["ISIN"]
            .astype(str)
            .str.strip()
            .str.upper()
            == str(row["ISIN"]).strip().upper()
        )

        matches = current[mask]

        if matches.empty:
            return None

        return matches.iloc[0]

    def determine_status(row):
        current_row = find_current_row(row)

        if current_row is None:
            return "NUOVO"

        for column in comparison_columns:
            bank_value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            current_value = pd.to_numeric(
                current_row[column],
                errors="coerce"
            )

            if pd.isna(bank_value) and pd.isna(current_value):
                continue

            if pd.isna(bank_value) or pd.isna(current_value):
                return "AGGIORNATO"

            if abs(bank_value - current_value) > tolerance:
                return "AGGIORNATO"

        return "INVARIATO"
    def get_changes(row):
        if row["Stato"] != "AGGIORNATO":
            return ""

        current_row = find_current_row(row)

        if current_row is None:
            return ""

        changes = []

        for column in comparison_columns:
            bank_value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            current_value = pd.to_numeric(
                current_row[column],
                errors="coerce"
            )

            if pd.isna(bank_value) and pd.isna(current_value):
                continue

            if pd.isna(bank_value) or pd.isna(current_value):
                changes.append(
                    f"{column}: {current_value} -> {bank_value}"
                )
                continue

            if abs(bank_value - current_value) > tolerance:
                changes.append(
                    f"{column}: {current_value} -> {bank_value}"
                )

        return " | ".join(changes)

    bank_certificates["Stato"] = bank_certificates.apply(
        determine_status,
        axis=1
    )
    bank_certificates["Variazioni"] = bank_certificates.apply(
        get_changes,
        axis=1
    )
    import_scopes = set(
        zip(
            bank_certificates["Titolare"].astype(str).str.strip().str.upper(),
            bank_certificates["Conto"].astype(str).str.strip().str.upper()
        )
    )

    missing_rows = []

    for _, current_row in current.iterrows():
        scope = (
            str(current_row["Titolare"]).strip().upper(),
            str(current_row["Conto"]).strip().upper()
        )

        key = (
            str(current_row["Titolare"]).strip().upper(),
            str(current_row["Conto"]).strip().upper(),
            str(current_row["ISIN"]).strip().upper()
        )

        if scope not in import_scopes:
            continue

        if key in bank_keys:
            continue

        missing_rows.append(
            {
                "Titolare": current_row["Titolare"],
                "Conto": current_row["Conto"],
                "ISIN": current_row["ISIN"],
                "Descrizione": current_row.get(
                    "Descrizione",
                    ""
                ),
                "Stato": "ASSENTE_DAL_NUOVO_ESTRATTO",
                "Variazioni": ""
            }
        )

    if missing_rows:
        missing_df = pd.DataFrame(
            missing_rows
        )

        bank_certificates = pd.concat(
            [
                bank_certificates,
                missing_df
            ],
            ignore_index=True,
            sort=False
        )

    return bank_certificates


def preview_fund_update(
    bank_df,
    current_funds
):
    """
    Confronta i fondi importati dalla banca con il
    portafoglio fondi attuale.

    La funzione produce solo una preview.
    Non modifica alcun file.
    """

    bank_funds = bank_df[
        bank_df["Tipo_Asset"] == "FONDO"
    ].copy()

    current = current_funds.copy()

    bank_keys = set(
        zip(
            bank_funds["Titolare"].astype(str).str.strip().str.upper(),
            bank_funds["Conto"].astype(str).str.strip().str.upper(),
            bank_funds["ISIN"].astype(str).str.strip().str.upper()
        )
    )

    comparison_columns = [
        "Quantita",
        "Prezzo_Carico",
        "Prezzo_Attuale",
        "Valore_Attuale",
        "Plusvalenza",
        "Plusvalenza_Percentuale"
    ]

    tolerance = 0.01

    def find_current_row(row):
        mask = (
            current["Titolare"]
            .astype(str)
            .str.strip()
            .str.upper()
            == str(row["Titolare"]).strip().upper()
        )

        mask &= (
            current["Conto"]
            .astype(str)
            .str.strip()
            .str.upper()
            == str(row["Conto"]).strip().upper()
        )

        mask &= (
            current["ISIN"]
            .astype(str)
            .str.strip()
            .str.upper()
            == str(row["ISIN"]).strip().upper()
        )

        matches = current[mask]

        if matches.empty:
            return None

        return matches.iloc[0]

    def determine_status(row):
        current_row = find_current_row(row)

        if current_row is None:
            return "NUOVO"

        for column in comparison_columns:
            bank_value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            current_value = pd.to_numeric(
                current_row[column],
                errors="coerce"
            )

            if pd.isna(bank_value) and pd.isna(current_value):
                continue

            if pd.isna(bank_value) or pd.isna(current_value):
                return "AGGIORNATO"

            if abs(bank_value - current_value) > tolerance:
                return "AGGIORNATO"

        return "INVARIATO"

    def get_changes(row):
        if row["Stato"] != "AGGIORNATO":
            return ""

        current_row = find_current_row(row)

        if current_row is None:
            return ""

        changes = []

        for column in comparison_columns:
            bank_value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            current_value = pd.to_numeric(
                current_row[column],
                errors="coerce"
            )

            if pd.isna(bank_value) and pd.isna(current_value):
                continue

            if pd.isna(bank_value) or pd.isna(current_value):
                changes.append(
                    f"{column}: {current_value} -> {bank_value}"
                )
                continue

            if abs(bank_value - current_value) > tolerance:
                changes.append(
                    f"{column}: {current_value} -> {bank_value}"
                )

        return " | ".join(changes)

    bank_funds["Stato"] = bank_funds.apply(
        determine_status,
        axis=1
    )

    bank_funds["Variazioni"] = bank_funds.apply(
        get_changes,
        axis=1
    )

    import_scopes = set(
        zip(
            bank_funds["Titolare"].astype(str).str.strip().str.upper(),
            bank_funds["Conto"].astype(str).str.strip().str.upper()
        )
    )

    missing_rows = []

    for _, current_row in current.iterrows():
        scope = (
            str(current_row["Titolare"]).strip().upper(),
            str(current_row["Conto"]).strip().upper()
        )

        key = (
            str(current_row["Titolare"]).strip().upper(),
            str(current_row["Conto"]).strip().upper(),
            str(current_row["ISIN"]).strip().upper()
        )

        if scope not in import_scopes:
            continue

        if key in bank_keys:
            continue

        missing_rows.append(
            {
                "Titolare": current_row["Titolare"],
                "Conto": current_row["Conto"],
                "ISIN": current_row["ISIN"],
                "Descrizione": current_row.get(
                    "Descrizione",
                    ""
                ),
                "Stato": "ASSENTE_DAL_NUOVO_ESTRATTO",
                "Variazioni": ""
            }
        )

    if missing_rows:
        missing_df = pd.DataFrame(
            missing_rows
        )

        bank_funds = pd.concat(
            [
                bank_funds,
                missing_df
            ],
            ignore_index=True,
            sort=False
        )

    return bank_funds


def preview_bank_import(
    bank_df,
    current_bonds,
    current_certificates,
    current_funds
):
    """
    Crea una preview unificata dell'import bancario.

    Riunisce:
    - bond;
    - certificate;
    - fondi;
    - eventuali strumenti da classificare.

    Non modifica alcun file.
    """

    bond_preview = preview_bond_update(
        bank_df,
        current_bonds
    )

    bond_preview["Tipo_Asset"] = "BOND"

    certificate_preview = preview_certificate_update(
        bank_df,
        current_certificates
    )

    certificate_preview["Tipo_Asset"] = "CERTIFICATE"

    fund_preview = preview_fund_update(
        bank_df,
        current_funds
    )

    fund_preview["Tipo_Asset"] = "FONDO"

    previews = [
        bond_preview,
        certificate_preview,
        fund_preview
    ]

    unknown = bank_df[
        bank_df["Tipo_Asset"] == "DA_CLASSIFICARE"
    ].copy()

    if not unknown.empty:
        unknown["Stato"] = "DA_CLASSIFICARE"
        unknown["Variazioni"] = ""
        previews.append(unknown)

    unified = pd.concat(
        previews,
        ignore_index=True,
        sort=False
    )

    preferred_columns = [
        "Titolare",
        "Conto",
        "Tipo_Asset",
        "ISIN",
        "Descrizione",
        "Quantita",
        "Prezzo_Carico",
        "Prezzo_Attuale",
        "Valore_Attuale",
        "Plusvalenza",
        "Plusvalenza_Percentuale",
        "Data_Riferimento",
        "Data_Importazione",
        "Stato",
        "Variazioni"
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in unified.columns
    ]

    return unified[available_columns]


def validate_bank_import_preview(preview):
    """
    Verifica se la preview dell'import bancario
    può procedere alla fase di conferma.

    Non modifica alcun file.
    """

    blocking_rows = preview[
        preview["Stato"] == "DA_CLASSIFICARE"
    ].copy()

    if not blocking_rows.empty:
        return {
            "ready": False,
            "message": (
                f"Importazione bloccata: "
                f"{len(blocking_rows)} strumento/i "
                f"da classificare."
            ),
            "blocking_rows": blocking_rows
        }

    return {
        "ready": True,
        "message": "Importazione pronta per la conferma.",
        "blocking_rows": blocking_rows
    }

def build_import_history_record(
    preview,
    source_file
):
    """
    Genera il record di storico relativo a una preview
    di importazione bancaria.

    Non scrive alcun file.
    """

    if preview.empty:
        raise ValueError(
            "Impossibile creare lo storico: preview vuota."
        )

    holders = (
        preview["Titolare"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    accounts = (
        preview["Conto"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    if len(holders) != 1:
        raise ValueError(
            "La preview deve contenere un solo titolare."
        )

    if len(accounts) != 1:
        raise ValueError(
            "La preview deve contenere un solo conto."
        )

    reference_dates = (
        pd.to_datetime(
            preview["Data_Riferimento"],
            errors="coerce"
        )
        .dropna()
        .dt.date
        .unique()
    )

    if len(reference_dates) != 1:
        raise ValueError(
            "La preview deve avere una sola Data_Riferimento."
        )

    import_dates = pd.to_datetime(
        preview["Data_Importazione"],
        errors="coerce"
    ).dropna()

    if import_dates.empty:
        raise ValueError(
            "Data_Importazione mancante."
        )

    import_timestamp = import_dates.max()

    import_id = (
        import_timestamp.strftime("%Y%m%d_%H%M%S_%f")
        + "_"
        + holders[0]
        + "_"
        + accounts[0]
    )

    import_id = (
        import_id
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    asset_counts = (
        preview["Tipo_Asset"]
        .value_counts()
        .to_dict()
    )

    validation = validate_bank_import_preview(
        preview
    )

    if validation["ready"]:
        status = "PRONTA"
    else:
        status = "BLOCCATA"

    record = {
        "ID_Importazione": import_id,
        "Data_Riferimento": reference_dates[0],
        "Data_Importazione": import_timestamp,
        "Titolare": holders[0],
        "Conto": accounts[0],
        "File_Origine": Path(source_file).name,
        "Totale_Strumenti": len(preview),
        "Numero_Bond": asset_counts.get("BOND", 0),
        "Numero_Certificate": asset_counts.get(
            "CERTIFICATE",
            0
        ),
        "Numero_Fondi": asset_counts.get("FONDO", 0),
        "Numero_Da_Classificare": asset_counts.get(
            "DA_CLASSIFICARE",
            0
        ),
        "Stato": status
    }

    return record


def save_import_history_record(
    record,
    history_file
):
    """
    Salva un record nello storico delle importazioni.

    Protezioni:
    - verifica l'esistenza del file;
    - verifica la struttura delle colonne;
    - impedisce ID_Importazione duplicati.
    """

    history_file = Path(history_file)

    if not history_file.exists():
        raise FileNotFoundError(
            f"File storico non trovato: {history_file}"
        )

    expected_columns = [
        "ID_Importazione",
        "Data_Riferimento",
        "Data_Importazione",
        "Titolare",
        "Conto",
        "File_Origine",
        "Totale_Strumenti",
        "Numero_Bond",
        "Numero_Certificate",
        "Numero_Fondi",
        "Numero_Da_Classificare",
        "Stato"
    ]

    history = pd.read_csv(history_file)

    if list(history.columns) != expected_columns:
        raise ValueError(
            "Struttura STORICO_IMPORTAZIONI.csv non valida."
        )

    missing_fields = [
        column
        for column in expected_columns
        if column not in record
    ]

    if missing_fields:
        raise ValueError(
            "Campi mancanti nel record storico: "
            + ", ".join(missing_fields)
        )

    import_id = str(
        record["ID_Importazione"]
    ).strip()

    if not history.empty:
        existing_ids = (
            history["ID_Importazione"]
            .astype(str)
            .str.strip()
        )

        if import_id in existing_ids.values:
            raise ValueError(
                "Importazione già presente nello storico: "
                + import_id
            )

    new_row = pd.DataFrame(
        [
            {
                column: record[column]
                for column in expected_columns
            }
        ]
    )

    updated_history = pd.concat(
        [
            history,
            new_row
        ],
        ignore_index=True
    )

    updated_history.to_csv(
        history_file,
        index=False
    )

    return updated_history
def prepare_bank_import(
    file_path,
    reference_date,
    holder,
    account,
    current_bonds,
    current_certificates,
    current_funds,
    bond_catalog=None,
    instruments=None
):
    """
    Prepara un'importazione bancaria completa.

    Esegue:
    - lettura e normalizzazione del CSV bancario;
    - classificazione degli strumenti;
    - generazione della preview unificata;
    - validazione della preview;
    - preparazione del record storico.

    Non modifica alcun file.
    """

    bank_df = read_bank_portfolio_csv(
        file_path=file_path,
        reference_date=reference_date,
        holder=holder,
        account=account
    )

    bank_df = classify_bank_assets(
        bank_df,
        bond_catalog=bond_catalog,
        certificates=current_certificates,
        instruments=instruments
    )

    preview = preview_bank_import(
        bank_df=bank_df,
        current_bonds=current_bonds,
        current_certificates=current_certificates,
        current_funds=current_funds
    )

    validation = validate_bank_import_preview(
        preview
    )

    history_record = build_import_history_record(
        preview=preview,
        source_file=file_path
    )

    return {
        "bank_data": bank_df,
        "preview": preview,
        "validation": validation,
        "history_record": history_record
    }


def create_portfolio_backup(
    files,
    backup_root
):
    """
    Crea una copia di sicurezza dei file indicati.

    Ogni esecuzione crea una cartella separata
    identificata dal timestamp.

    Non modifica i file originali.
    """

    backup_root = Path(backup_root)

    timestamp = (
        pd.Timestamp.now(tz="Europe/Rome")
        .strftime("%Y%m%d_%H%M%S_%f")
    )

    backup_dir = (
        backup_root
        / f"backup_{timestamp}"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    created_backups = []

    try:
        for file_path in files:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(
                    f"File da salvare non trovato: {file_path}"
                )

            destination = (
                backup_dir
                / file_path.name
            )

            shutil.copy2(
                file_path,
                destination
            )

            created_backups.append(
                destination
            )

    except Exception:
        shutil.rmtree(
            backup_dir,
            ignore_errors=True
        )
        raise

    return {
        "backup_dir": backup_dir,
        "files": created_backups
    }