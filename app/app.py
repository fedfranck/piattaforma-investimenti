
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
from calcoli import (
    enrich_bonds, enrich_certificates, portfolio_kpis,
    coupon_forecast, bond_rotation_candidates, certificate_alerts
)
from reinvestment import reinvestment_candidates

st.set_page_config(page_title="Piattaforma Investimenti", page_icon="📊", layout="wide")

def load_data():
    return load_portfolio_data(ROOT / "data", ROOT)

bonds, certs, underlyings, coupons, catalog, liquidity, holders = load_data()

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
        st.sidebar.error(f"Errore nell'importazione: {exc}")
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

        if suggestions.empty:
            st.warning(
                "Nessun titolo disponibile nel catalogo."
            )
        else:
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
