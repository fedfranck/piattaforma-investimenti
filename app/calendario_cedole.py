"""Motore calendario cedolare V3.

Genera i flussi cedolari previsti delle obbligazioni in portafoglio.
Mantiene distinte le posizioni per titolare e conto.
Le date e le aliquote fiscali devono essere verificate prima
di considerare definitivi gli importi netti.
"""

from datetime import date

import pandas as pd


FREQUENZE = {
    "Annuale": 12,
    "Semestrale": 6,
}


def genera_calendario_cedole(portafoglio, anagrafica, data_inizio=None, mesi=12):
    """Genera i flussi previsti, conservando le singole posizioni."""
    if not isinstance(mesi, int) or mesi <= 0:
        raise ValueError("Il numero di mesi deve essere positivo")

    colonne = [
        "Data", "ISIN", "Titolare", "Conto", "Descrizione",
        "Nominale", "Cedola_Lorda", "Cedola_Netta_Stimata",
        "Stato", "Fonte"
    ]

    if portafoglio.empty:
        return pd.DataFrame(columns=colonne)

    inizio = pd.Timestamp(data_inizio if data_inizio is not None else date.today())
    fine = inizio + pd.DateOffset(months=mesi)

    if anagrafica["ISIN"].duplicated().any():
        raise ValueError("ISIN duplicati nell'anagrafica")

    campi = [
        "ISIN", "Cedola_Percentuale", "Frequenza_Cedola",
        "Data_Prossima_Cedola", "Data_Scadenza", "Fonte", "Aliquota_Fiscale"
    ]
    dati = portafoglio.merge(
        anagrafica[campi],
        on="ISIN",
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("_Portafoglio", "")
    )

    if (dati["_merge"] != "both").any():
        mancanti = dati.loc[dati["_merge"] != "both", "ISIN"].tolist()
        raise ValueError(f"ISIN senza anagrafica: {mancanti}")

    righe = []
    for _, r in dati.iterrows():
        scadenza = pd.to_datetime(r["Data_Scadenza"], errors="coerce")
        if pd.isna(scadenza):
            raise ValueError(f"Scadenza non valida: {r['ISIN']}")

        date_cedole = genera_date_cedole(
            r["Data_Prossima_Cedola"],
            r["Frequenza_Cedola"],
            inizio,
            min(fine, scadenza)
        )

        aliquota = r["Aliquota_Fiscale"]
        if pd.notna(aliquota):
            aliquota = float(aliquota)
            if not 0 <= aliquota <= 1:
                raise ValueError(f"Aliquota fiscale non valida: {r['ISIN']}")
        else:
            aliquota = None

        importi = calcola_importo_cedola(
            r["Quantita_Nominale"],
            r["Cedola_Percentuale"],
            r["Frequenza_Cedola"],
            aliquota=aliquota
        )

        fonte = str(r.get("Fonte", ""))
        stato = "DA VERIFICARE" if (
            "verificare" in fonte.lower()
            or "confermare" in fonte.lower()
        ) else "PREVISTA"

        for giorno in date_cedole:
            righe.append({
                "Data": giorno,
                "ISIN": r["ISIN"],
                "Titolare": r.get("Titolare"),
                "Conto": r.get("Conto"),
                "Descrizione": r.get("Descrizione"),
                "Nominale": r["Quantita_Nominale"],
                "Cedola_Lorda": importi["Lordo"],
                "Cedola_Netta_Stimata": importi["Netto_Stimato"],
                "Stato": stato,
                "Fonte": fonte
            })

    return pd.DataFrame(righe, columns=colonne).sort_values(
        "Data"
    ).reset_index(drop=True)


def genera_date_cedole(prossima_cedola, frequenza, data_inizio, data_fine):
    """Restituisce le ricorrenze cedolari comprese nel periodo richiesto.

    Le date sono contrattuali/previste: non applica aggiustamenti
    per giorni festivi o calendari bancari.
    """
    if frequenza not in FREQUENZE:
        raise ValueError(f"Frequenza cedolare non riconosciuta: {frequenza}")

    data = pd.to_datetime(prossima_cedola, errors="coerce")
    inizio = pd.to_datetime(data_inizio)
    fine = pd.to_datetime(data_fine)

    if pd.isna(data):
        raise ValueError("Data prossima cedola mancante o non valida")
    if fine < inizio:
        raise ValueError("La data finale precede quella iniziale")

    passo = pd.DateOffset(months=FREQUENZE[frequenza])
    date = []

    while data < inizio:
        data += passo

    while data <= fine:
        date.append(data)
        data += passo

    return date


def calcola_importo_cedola(nominale, tasso_annuo, frequenza, aliquota=None):
    """Calcola gli importi per una singola cedola.

    aliquota è espressa come numero decimale (es. 0.125).
    Se non specificata, il netto resta non disponibile.
    """
    if frequenza not in FREQUENZE:
        raise ValueError(f"Frequenza non riconosciuta: {frequenza}")

    nominale = float(nominale)
    tasso_annuo = float(tasso_annuo)

    if nominale < 0 or tasso_annuo < 0:
        raise ValueError("Nominale e tasso non possono essere negativi")

    pagamenti_annui = 12 // FREQUENZE[frequenza]
    lordo = nominale * tasso_annuo / 100 / pagamenti_annui

    if aliquota is None:
        return {"Lordo": round(lordo, 2), "Netto_Stimato": None}

    aliquota = float(aliquota)
    if not 0 <= aliquota <= 1:
        raise ValueError("Aliquota non valida")

    netto = lordo * (1 - aliquota)
    return {
        "Lordo": round(lordo, 2),
        "Netto_Stimato": round(netto, 2),
    }


def riepilogo_mensile_cedole(calendario, data_inizio, mesi=12):
    """Riepilogo mensile lordo per titolare, inclusi i mesi senza cedole."""
    if not isinstance(mesi, int) or mesi <= 0:
        raise ValueError("Il numero di mesi deve essere positivo")

    inizio = pd.Timestamp(data_inizio)
    fine = inizio + pd.DateOffset(months=mesi)
    indice = pd.period_range(
        inizio.to_period("M"), fine.to_period("M"), freq="M"
    )

    if calendario.empty:
        return pd.DataFrame({"TOTALE": 0.0}, index=indice)

    dati = calendario.copy()
    dati["Data"] = pd.to_datetime(dati["Data"])
    dati = dati.loc[
        (dati["Data"] >= inizio) & (dati["Data"] <= fine)
    ].copy()
    dati["Mese"] = dati["Data"].dt.to_period("M")

    tabella = dati.pivot_table(
        index="Mese",
        columns="Titolare",
        values="Cedola_Lorda",
        aggfunc="sum",
        fill_value=0
    )

    tabella = tabella.reindex(indice, fill_value=0)
    tabella["TOTALE"] = tabella.sum(axis=1)

    return tabella.round(2)
