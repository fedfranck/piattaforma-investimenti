
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from database import load_portfolio_data, sync_to_sqlite

from bank_import import (
    prepare_complete_bank_import,
    execute_prepared_bank_import
)

from calcoli import (
    enrich_bonds, enrich_certificates, portfolio_kpis,
    coupon_forecast, bond_rotation_candidates, certificate_alerts
)
from reinvestment import (
    reinvestment_candidates,
    build_reinvestment_allocation
)

st.set_page_config(page_title="Piattaforma Investimenti", page_icon="📊", layout="wide")

def load_data():
    return load_portfolio_data(ROOT / "data", ROOT)

bonds, certs, underlyings, coupons, catalog, liquidity, holders = load_data()

# Copia del database Bond con lo schema originale.
# Viene usata dall'importatore bancario Level 2 prima
# dell'arricchimento grafico con la colonna Emittente.
bank_import_bonds = bonds.copy()

# ============================================================
# ARCHIVI AGGIUNTIVI PER IMPORTAZIONE BANCARIA LEVEL 2
# ============================================================

funds = pd.read_csv(
    ROOT / "data" / "FONDI_PORTAFOGLIO.csv"
)

etfs = pd.read_csv(
    ROOT / "data" / "ETF_PORTAFOGLIO.csv"
)

stocks = pd.read_csv(
    ROOT / "data" / "AZIONI_PORTAFOGLIO.csv"
)

import_history = pd.read_csv(
    ROOT / "data" / "STORICO_IMPORTAZIONI.csv"
)

instruments = pd.read_csv(
    ROOT / "data" / "STRUMENTI.csv",
    sep=";"
)

# Completa il portafoglio Bond con l'emittente presente nel catalogo
if "Emittente" not in bonds.columns and not catalog.empty:
    if "ISIN" in bonds.columns and "ISIN" in catalog.columns and "Emittente" in catalog.columns:
        issuer_map = (
            catalog[["ISIN", "Emittente"]]
            .drop_duplicates("ISIN")
        )

        bonds = bonds.merge(
            issuer_map,
            on="ISIN",
            how="left"
        )

st.title("📊 Piattaforma Investimenti")
st.caption("V1 operativa — dati locali CSV, calcoli automatici e motore di rotazione supervisionato")

# ============================================================
# IMPORTAZIONE / AGGIORNAMENTO CSV
# ============================================================

st.sidebar.divider()
st.sidebar.subheader("📥 Aggiornamento dati")

csv_files = {
    "Bond portafoglio": ROOT / "data" / "BOND_PORTAFOGLIO.csv",
    "Certificates portafoglio": ROOT / "data" / "CERTIFICATES_PORTAFOGLIO.csv",
    "Sottostanti certificates": ROOT / "data" / "SOTTOSTANTI_CERTIFICATES.csv",
    "Calendario cedole": ROOT / "data" / "CALENDARIO_CEDOLE.csv",
    "Catalogo bond": ROOT / "data" / "CATALOGO_BOND.csv",
    "Liquidità": ROOT / "LIQUIDITA.csv",
    "Titolari": ROOT / "data" / "TITOLARI.csv",
}

required_columns = {
    "Bond portafoglio": [
        "Titolare",
        "ISIN",
        "Quantita_Nominale",
        "Prezzo_Attuale",
    ],
    "Certificates portafoglio": [
        "ISIN",
        "Descrizione",
        "Quantita",
        "Prezzo_Attuale",
    ],
    "Sottostanti certificates": [
        "ISIN_CERTIFICATE",
        "Sottostante",
        "Ticker",
        "Strike",
        "Barriera",
    ],
    "Calendario cedole": [
        "ISIN",
        "Data_Cedola",
        "Importo_Lordo",
    ],
    "Catalogo bond": [
        "ISIN",
        "Descrizione",
        "Emittente",
        "Prezzo",
        "Yield",
    ],
    "Liquidità": [
        "Titolare",
        "Conto",
        "Data",
        "Saldo",
    ],
    "Titolari": [
        "Titolare",
        "Banca",
    ],
}

selected_csv = st.sidebar.selectbox(
    "Seleziona archivio da aggiornare",
    list(csv_files.keys())
)

uploaded_csv = st.sidebar.file_uploader(
    "Scegli il nuovo file CSV",
    type=["csv"]
)

if uploaded_csv is not None:
    try:
        preview_df = pd.read_csv(uploaded_csv)

        missing_columns = [
            col
            for col in required_columns[selected_csv]
            if col not in preview_df.columns
        ]

        if missing_columns:
            st.sidebar.error(
                "❌ File non valido. Mancano le colonne: "
                + ", ".join(missing_columns)
            )
        else:
            st.sidebar.success(
                f"File valido: {uploaded_csv.name} — "
                f"{len(preview_df)} righe"
            )

            st.sidebar.dataframe(
                preview_df.head(5),
                use_container_width=True
            )

            if st.sidebar.button("✅ Conferma aggiornamento"):
                destination = csv_files[selected_csv]

                # Backup del file precedente
                if destination.exists():
                    backup_dir = destination.parent / "backup"
                    backup_dir.mkdir(exist_ok=True)

                    from datetime import datetime
                    import shutil

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    backup_file = (
                        backup_dir
                        / f"{destination.stem}_backup_{timestamp}.csv"
                    )

                    shutil.copy2(destination, backup_file)

                # Salva il nuovo CSV
                preview_df.to_csv(destination, index=False)

                # Aggiorna i dati caricati dalla cache
                load_data.clear()

                st.sidebar.success(
                    f"{selected_csv} aggiornato correttamente."
                )

                st.rerun()

    except Exception as exc:
        st.sidebar.error(f"Errore nell'importazione: {exc}"
        )

if "bank_import_success" in st.session_state:
    st.sidebar.success(
        st.session_state.pop(
            "bank_import_success"
        )
    )


# ============================================================
# IMPORTAZIONE PORTAFOGLIO BANCA - LEVEL 2
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "🏦 Importa portafoglio banca"
)

st.sidebar.caption(
    "Importazione automatica dell'estratto titoli "
    "con preview e conferma prima dell'aggiornamento."
)

bank_uploaded_file = st.sidebar.file_uploader(
    "Carica I_miei_titoli.csv",
    type=["csv"],
    key="bank_portfolio_uploader"
)
bank_holder_options = []

if (
    not holders.empty
    and "Titolare" in holders.columns
):
    bank_holder_options = sorted(
        holders["Titolare"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

bank_holder = st.sidebar.selectbox(
    "Titolare del portafoglio",
    bank_holder_options,
    key="bank_import_holder"
)

bank_account = st.sidebar.text_input(
    "Conto / Banca",
    value="Mediobanca",
    key="bank_import_account"
)

bank_reference_date = st.sidebar.date_input(
    "Data di riferimento",
    key="bank_import_reference_date"
)

if st.sidebar.button(
    "🔍 Analizza portafoglio",
    key="bank_import_analyze"
):

    if bank_uploaded_file is None:

        st.sidebar.error(
            "Carica prima il file I_miei_titoli.csv."
        )

    elif not bank_holder:

        st.sidebar.error(
            "Seleziona il titolare del portafoglio."
        )

    elif not bank_account.strip():

        st.sidebar.error(
            "Inserisci il conto o la banca."
        )

    else:

        try:

            # ------------------------------------------
            # Salva temporaneamente il file caricato
            # ------------------------------------------

            import_dir = (
                ROOT / "import"
            )

            import_dir.mkdir(
                exist_ok=True
            )

            uploaded_bank_path = (
                import_dir
                / "I_miei_titoli_streamlit.csv"
            )

            uploaded_bank_path.write_bytes(
                bank_uploaded_file.getvalue()
            )

            # ------------------------------------------
            # File reali gestiti dalla transazione
            # ------------------------------------------

            portfolio_target_files = {
                "bonds":
                    ROOT / "data"
                    / "BOND_PORTAFOGLIO.csv",

                "certificates":
                    ROOT / "data"
                    / "CERTIFICATES_PORTAFOGLIO.csv",

                "funds":
                    ROOT / "data"
                    / "FONDI_PORTAFOGLIO.csv",

                "etfs":
                    ROOT / "data"
                    / "ETF_PORTAFOGLIO.csv",

                "stocks":
                    ROOT / "data"
                    / "AZIONI_PORTAFOGLIO.csv"
            }

            history_file = (
                ROOT / "data"
                / "STORICO_IMPORTAZIONI.csv"
            )

            temp_root = (
                ROOT / "import" / "temp"
            )

            temp_root.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------
            # FASE A: preparazione SENZA scrittura
            # ------------------------------------------

            bank_preparation = (
                prepare_complete_bank_import(
                    file_path=uploaded_bank_path,
                    reference_date=(
                        bank_reference_date.isoformat()
                    ),
                    holder=bank_holder,
                    account=bank_account.strip(),
                    current_bonds=bank_import_bonds,
                    current_certificates=certs,
                    current_funds=funds,
                    current_etfs=etfs,
                    current_stocks=stocks,
                    current_history=import_history,
                    portfolio_target_files=(
                        portfolio_target_files
                    ),
                    history_file=history_file,
                    temp_root=temp_root,
                    bond_catalog=catalog,
                    instruments=instruments
                )
            )

            # ------------------------------------------
            # Conserva la preparazione tra i rerun
            # ------------------------------------------

            st.session_state[
                "bank_import_preparation"
            ] = bank_preparation

            st.session_state[
                "bank_import_target_files"
            ] = {
                **portfolio_target_files,
                "history": history_file
            }

            st.sidebar.success(
                "Analisi completata. "
                "Nessun file reale è stato modificato."
            )

        except Exception as exc:

            st.session_state.pop(
                "bank_import_preparation",
                None
            )

            st.session_state.pop(
                "bank_import_target_files",
                None
            )

            st.sidebar.error(
                f"Errore nell'analisi: {exc}"
            )

# ============================================================
# PREVIEW IMPORTAZIONE BANCARIA - LEVEL 2
# ============================================================

if "bank_import_preparation" in st.session_state:

    bank_preparation = st.session_state[
        "bank_import_preparation"
    ]

    bank_preview = bank_preparation[
        "preview"
    ]

    bank_validation = bank_preparation[
        "validation"
    ]

    st.sidebar.divider()

    st.sidebar.markdown(
        "### 🔎 Anteprima importazione"
    )

    # --------------------------------------------------------
    # Conteggio strumenti per asset class
    # --------------------------------------------------------

    asset_counts = (
        bank_preview["Tipo_Asset"]
        .value_counts()
        .to_dict()
    )

    st.sidebar.write(
        f"**Bond:** "
        f"{asset_counts.get('BOND', 0)}"
    )

    st.sidebar.write(
        f"**Certificate:** "
        f"{asset_counts.get('CERTIFICATE', 0)}"
    )

    st.sidebar.write(
        f"**Fondi:** "
        f"{asset_counts.get('FONDO', 0)}"
    )

    st.sidebar.write(
        f"**ETF:** "
        f"{asset_counts.get('ETF', 0)}"
    )

    st.sidebar.write(
        f"**Azioni:** "
        f"{asset_counts.get('AZIONE', 0)}"
    )

    st.sidebar.write(
        f"**Da classificare:** "
        f"{asset_counts.get('DA_CLASSIFICARE', 0)}"
    )

    # --------------------------------------------------------
    # Conteggio variazioni
    # --------------------------------------------------------

    if "Stato" in bank_preview.columns:

        status_counts = (
            bank_preview["Stato"]
            .value_counts()
            .to_dict()
        )

        st.sidebar.markdown(
            "**Variazioni rilevate**"
        )

        st.sidebar.write(
            f"Nuovi: "
            f"{status_counts.get('NUOVO', 0)}"
        )

        st.sidebar.write(
            f"Aggiornati: "
            f"{status_counts.get('AGGIORNATO', 0)}"
        )

        st.sidebar.write(
            f"Invariati: "
            f"{status_counts.get('INVARIATO', 0)}"
        )

        st.sidebar.write(
            "Assenti dal nuovo estratto: "
            f"{status_counts.get(
                'ASSENTE_DAL_NUOVO_ESTRATTO',
                0
            )}"
        )

    # --------------------------------------------------------
    # Stato validazione
    # --------------------------------------------------------

    if bank_validation.get(
        "ready",
        False
    ):

        st.sidebar.success(
            "Importazione pronta per la conferma."
        )

    else:

        st.sidebar.error(
            bank_validation.get(
                "message",
                "Importazione non valida."
            )
        )

    # --------------------------------------------------------
    # Tabella dettagliata
    # --------------------------------------------------------

    preview_columns = [
        column
        for column in [
            "Tipo_Asset",
            "ISIN",
            "Descrizione",
            "Stato",
            "Variazioni"
        ]
        if column in bank_preview.columns
    ]

    st.sidebar.dataframe(
        bank_preview[
            preview_columns
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Conferma aggiornamento reale
    # --------------------------------------------------------

    st.sidebar.divider()

    bank_import_confirmed = st.sidebar.checkbox(
        "Confermo di voler aggiornare il portafoglio",
        key="bank_import_confirmation"
    )

    confirm_bank_import = st.sidebar.button(
        "✅ Conferma aggiornamento portafoglio",
        key="bank_import_execute",
        disabled=(
            not bank_validation.get("ready", False)
            or not bank_import_confirmed
        )
    )

    if confirm_bank_import:

        try:

            transaction_result = (
                execute_prepared_bank_import(
                    preparation=bank_preparation,
                    target_files=st.session_state[
                        "bank_import_target_files"
                    ],
                    backup_root=(
                        ROOT
                        / "data"
                        / "backup_bank_import"
                    ),
                    cleanup_temp=True
                )
            )

            # La transazione è terminata correttamente.
            # Eliminiamo la preparazione dalla sessione
            # per impedire una seconda esecuzione accidentale.

            st.session_state.pop(
                "bank_import_preparation",
                None
            )

            st.session_state.pop(
                "bank_import_target_files",
                None
            )

            st.session_state[
                "bank_import_success"
            ] = (
                "Portafoglio aggiornato correttamente. "
                "Backup creato prima dell'aggiornamento."
            )

            st.rerun()

        except Exception as exc:

            st.sidebar.error(
                "Aggiornamento non completato: "
                f"{exc}"
            )

holder_options = ["Tutti"]

if not holders.empty and "Titolare" in holders.columns:
    holder_options += sorted(
        holders["Titolare"]
        .dropna()
        .astype(str)
        .unique()
    .tolist()
    )
else:
    holder_options += sorted(set(
        list(bonds.get("Titolare", pd.Series(dtype=str)).dropna().astype(str))
        + list(liquidity.get("Titolare", pd.Series(dtype=str)).dropna().astype(str))
    ))
selected_holder = st.sidebar.selectbox("Titolare", holder_options)

if selected_holder != "Tutti":
    if "Titolare" in bonds: bonds = bonds[bonds["Titolare"].astype(str) == selected_holder].copy()
    if "Titolare" in liquidity: liquidity = liquidity[liquidity["Titolare"].astype(str) == selected_holder].copy()

bonds = enrich_bonds(bonds)

# Classificazione delle scadenze per analisi della concentrazione temporale
if not bonds.empty and "Giorni_Scadenza" in bonds.columns:
    bonds["Fascia_Scadenza"] = np.select(
        [
            bonds["Giorni_Scadenza"] < 730,
            bonds["Giorni_Scadenza"] < 1825,
            bonds["Giorni_Scadenza"] < 3650
        ],
        [
            "< 2 anni",
            "2–5 anni",
            "5–10 anni"
        ],
        default="> 10 anni"
    )

# Percentuale di ogni posizione sul portafoglio Bond
if not bonds.empty and "Valore_Attuale_Calc" in bonds.columns:
    if bonds["Valore_Attuale_Calc"].sum() > 0:
        bonds["% Portafoglio Bond"] = (
            bonds["Valore_Attuale_Calc"]
            / bonds["Valore_Attuale_Calc"].sum()
            * 100
        )
    else:
        bonds["% Portafoglio Bond"] = 0

certs = enrich_certificates(certs, underlyings)
kpis = portfolio_kpis(bonds, certs, liquidity)
# Keep the operational SQLite snapshot aligned with the current CSV sources.
try:
    sync_to_sqlite(ROOT, bonds, certs, underlyings, coupons)
except Exception:
    pass
forecast = coupon_forecast(bonds, coupons)

st.sidebar.divider()
st.sidebar.metric("Posizioni Bond", len(bonds))
st.sidebar.metric("Certificates", len(certs))
if not certs.empty and "Classificazione" in certs:
    st.sidebar.metric("Alert Certificates", int((certs["Classificazione"] == "CRITICO").sum()))
c1,c2,c3,c4,c5 = st.columns(5)

c1.metric(
    "Patrimonio",
    f"€ {kpis['patrimonio']:,.2f}"
)

c2.metric(
    "Bond",
    f"€ {kpis['bond_value']:,.2f}"
)

c3.metric(
    "Certificates",
    f"€ {kpis['cert_value']:,.2f}"
)

c4.metric(
    "Liquidità",
    f"€ {kpis['liquidity']:,.2f}"
)

c5.metric(
    "Plus/Minus",
    f"€ {kpis['unrealized_pnl']:,.2f}",
    f"{kpis['rendimento_totale_pct']:.2f}%"
)
st.markdown("### 📈 Rendimento e Cash Flow")

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Capitale investito",
    f"€ {kpis['capitale_investito']:,.2f}"
)

r2.metric(
    "Cedole annue",
    f"€ {kpis['annual_coupon']:,.2f}"
)

r3.metric(
    "Yield cedolare",
    f"{kpis['rendimento_cedolare_capitale_pct']:.2f}%"
)

r4.metric(
    "Cash Flow medio mensile",
    f"€ {kpis['annual_coupon'] / 12:,.2f}"
)

st.divider()
t1,t2,t3,t4,t5,t6 = st.tabs(["Dashboard","Bond","Certificates","Cedole","Rotazioni","Dati"])

with t1:
    a, b, c = st.columns(3)
    a.metric(
        "Flusso cedolare annuo Bond",
        f"€ {kpis['annual_coupon']:,.2f}"
    )
    b.metric(
        "Cedole incassate",
        f"€ {kpis['coupons_received']:,.2f}"
    )
    c.metric(
        "Flusso residuo stimato",
        f"€ {max(kpis['annual_coupon'] - kpis['coupons_received'], 0):,.2f}"
    )

    st.subheader("Asset allocation")

    alloc = pd.DataFrame({
        "Asset": ["Bond", "Certificates", "Liquidità"],
        "Valore": [
            kpis["bond_value"],
            kpis["cert_value"],
            kpis["liquidity"]
        ]
    })

    if kpis["patrimonio"] > 0:
        alloc["% Patrimonio"] = (
            alloc["Valore"] / kpis["patrimonio"] * 100
        )
    else:
        alloc["% Patrimonio"] = 0

    st.bar_chart(alloc.set_index("Asset"))

    alloc_display = alloc.copy()
    alloc_display["Valore"] = alloc_display["Valore"].map(
        lambda x: f"€ {x:,.2f}"
    )
    alloc_display["% Patrimonio"] = alloc_display["% Patrimonio"].map(
        lambda x: f"{x:.2f}%"
    )

    st.dataframe(
        alloc_display,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Prossime cedole")
    if forecast.empty:
        st.info(
            "Nessuna cedola futura disponibile nel calendario. "
            "Le date presenti nei Bond possono essere usate "
            "per generare il calendario."
        )
    else:
        st.dataframe(
            forecast.head(20),
            use_container_width=True
        )

    st.subheader("Alert")
    for x in bond_rotation_candidates(bonds):
        st.warning(
            f"Bond candidato alla rotazione: "
            f"{x['ISIN']} — score {x['score']}"
        )
    for x in certificate_alerts(certs):
        st.error(f"Certificate: {x}")
with t2:
    st.subheader("Portafoglio Bond")

    if bonds.empty:
        st.info("Nessun Bond presente.")
    else:
        cols = [
            "ISIN",
            "Descrizione",
            "Valore_Carico",
            "Valore_Attuale",
            "Plusvalenza",
            "Plusvalenza_Percentuale",
            "Cedola_Percentuale",
            "Cedola_Netta_Annua",
            "Yield_On_Cost",
            "Giorni_Scadenza",
            "Score_Uscita",
            "Classificazione"
        ]

        st.dataframe(
            bonds[[c for c in cols if c in bonds.columns]],
            use_container_width=True,
            hide_index=True
        )

st.subheader("Dettaglio scoring")

if not bonds.empty:
    scoring_cols = [
    "ISIN",
    "Plusvalenza_Percentuale",
    "Giorni_Scadenza",
    "Yield_On_Cost",
    "Score_Plusvalenza",
    "Score_Scadenza",
    "Score_Yield",
    "Score_Uscita",
    "Classificazione",
    "% Portafoglio Bond"
]

    st.dataframe(
        bonds[
            [c for c in scoring_cols if c in bonds.columns]
        ],
        use_container_width=True,
        hide_index=True
    )

st.subheader("Concentrazione per emittente")

if "Emittente" in bonds.columns and not bonds.empty:
    issuer_analysis = (
        bonds.groupby(
            "Emittente",
            as_index=False
        )["Valore_Attuale_Calc"]
        .sum()
        .rename(
            columns={
                "Valore_Attuale_Calc": "Valore"
            }
        )
    )

    if kpis["bond_value"] > 0:
        issuer_analysis["% Bond"] = (
            issuer_analysis["Valore"]
            / kpis["bond_value"]
            * 100
        )
    else:
        issuer_analysis["% Bond"] = 0

    issuer_analysis = issuer_analysis.sort_values(
        "Valore",
        ascending=False
    )

    st.dataframe(
        issuer_analysis,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Dati emittente non disponibili.")
with t3:
    st.subheader("Certificates")
    if certs.empty:
        st.info("Il file data/CERTIFICATES_PORTAFOGLIO.csv è attualmente vuoto. Inserisci le posizioni per attivare l'analisi.")
    else:
        cols = ["ISIN","Descrizione","Prezzo_Attuale","Worst_Of","Distanza_Barriera_Cedolare",
                "Distanza_Barriera_Capitale","Classificazione"]
        st.dataframe(certs[[c for c in cols if c in certs.columns]], use_container_width=True, hide_index=True)
with t4:
    st.subheader("Flussi cedolari")

    st.metric("Flusso annuo teorico", f"€ {kpis['annual_coupon']:,.2f}")

    if not forecast.empty:
        forecast2 = forecast.copy()
        ...
    else:
        st.info("Calendario cedole non ancora valorizzato.")

    st.divider()
    st.subheader("Simulazione reinvestimento cedole")

    profile_options = {
        "💰 Massimo Cash Flow": "cash_flow",
        "📈 Massimo rendimento": "rendimento",
        "🪜 Ladder / Diversificazione": "ladder"
    }

    selected_profile_label = st.selectbox(
        "Profilo reinvestimento",
        list(profile_options.keys())
    )

    selected_profile = profile_options[selected_profile_label]

    reinvest_amount = st.number_input(
        "Importo disponibile da reinvestire (€)",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )
    if st.button("Calcola opportunità reinvestimento"):
        suggestions = reinvestment_candidates(
            catalog,
            reinvest_amount,
            bonds,
            profile=selected_profile
        )

        allocation, residual = build_reinvestment_allocation(
            catalog,
            reinvest_amount,
            bonds=bonds,
            profile=selected_profile,
            max_positions=3
        )

        if suggestions.empty:
            st.warning(
                "Nessun titolo disponibile nel catalogo."
            )
        else:
            st.subheader("Allocazione proposta del capitale")

            if allocation.empty:
                st.info(
                    "Non è stato possibile costruire un'allocazione "
                    "con l'importo disponibile."
                )
            else:
                allocation_display = allocation[
                    [
                        "ISIN",
                        "Descrizione",
                        "Prezzo",
                        "Lotti_Allocati",
                        "Nominale_Allocato",
                        "Capitale_Investito"
                    ]
                ].copy()

                st.dataframe(
                    allocation_display,
                    use_container_width=True,
                    hide_index=True
                )

                totale_investito = allocation[
                    "Capitale_Investito"
                ].sum()

                nominale_totale = allocation[
                    "Nominale_Allocato"
                ].sum()

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Capitale disponibile",
                    f"€ {reinvest_amount:,.2f}"
                )

                col2.metric(
                    "Capitale investito",
                    f"€ {totale_investito:,.2f}"
                )

                col3.metric(
                    "Liquidità residua",
                    f"€ {residual:,.2f}"
                )

                col4.metric(
                    "Nominale totale",
                    f"€ {nominale_totale:,.0f}"
                )

            st.divider()
            display_columns = [
                "ISIN",
                "Descrizione",
                "Prezzo",
                "Cedola",
                "Yield",
                "Duration",
                "Scadenza",
                "Rating",
                "CashFlow_Annuale",
                "Score_Yield",
                "Score_CashFlow",
                "Score_Diversificazione",
                "Score_Rating",
                "Score_Prezzo",
                "Contributo_Yield",
                "Contributo_CashFlow",
                "Contributo_Diversificazione",
                "Contributo_Rating",
                "Contributo_Prezzo",
                "Score_Reinvestimento",
                "Motivazione"
            ]

            st.dataframe(
                suggestions[
                    [c for c in display_columns if c in suggestions.columns]
                ],
                use_container_width=True,
                hide_index=True
            )
with t5:
    st.subheader("Proposte di rotazione Bond")

    proposals = bond_rotation_candidates(bonds, catalog)

    if not proposals:
        st.success(
            "Nessuna proposta automatica sulla base delle regole attuali."
        )
    else:
        proposal_rows = []

        for proposal in proposals:
            isin = proposal.get("ISIN")

            match = bonds[bonds["ISIN"].astype(str) == str(isin)]

            if not match.empty:
                row = match.iloc[0]

                proposal_rows.append({
                    "ISIN": isin,
                    "Score": proposal.get("score"),
                    "Classificazione": row.get("Classificazione"),
                    "% Portafoglio Bond": row.get("% Portafoglio Bond"),
                    "Plusvalenza %": row.get("Plusvalenza_Percentuale"),
                    "Giorni Scadenza": row.get("Giorni_Scadenza"),
                    "Yield On Cost": row.get("Yield_On_Cost")
                })
            else:
                proposal_rows.append(proposal)

        p = pd.DataFrame(proposal_rows)

        st.dataframe(
            p,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Le proposte sono segnali quantitativi da verificare "
            "prima di qualsiasi operazione."
        )
with t6:
    st.subheader("Qualità e stato dei dati")
    status = [
        ("Bond", len(bonds), "OK" if len(bonds) else "VUOTO"),
        ("Certificates", len(certs), "OK" if len(certs) else "VUOTO"),
        ("Sottostanti", len(underlyings), "OK" if len(underlyings) else "VUOTO"),
        ("Calendario cedole", len(coupons), "OK" if len(coupons) else "VUOTO"),
        ("Catalogo Bond", len(catalog), "OK" if len(catalog) else "VUOTO"),
    ]
    st.dataframe(pd.DataFrame(status, columns=["Dataset","Righe","Stato"]), use_container_width=True, hide_index=True)

    st.subheader("CSV caricati")
    for name, df in [("BOND_PORTAFOGLIO",bonds),("CERTIFICATES_PORTAFOGLIO",certs),
                     ("SOTTOSTANTI_CERTIFICATES",underlyings),("CALENDARIO_CEDOLE",coupons),
                     ("CATALOGO_BOND",catalog),("LIQUIDITA",liquidity)]:
        with st.expander(name):
            st.dataframe(df, use_container_width=True, hide_index=True)
