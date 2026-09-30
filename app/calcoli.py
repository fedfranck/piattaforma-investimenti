
from datetime import date
import numpy as np
import pandas as pd

def _num(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)

def enrich_bonds(df):
    df=df.copy()
    if df.empty: return df
    q=_num(df,"Quantita_Nominale")
    pc=_num(df,"Prezzo_Carico")
    pa=_num(df,"Prezzo_Attuale")
    df["Valore_Carico_Calc"]=q*pc/100
    df["Valore_Attuale_Calc"]=q*pa/100
    df["Plusvalenza_Calc"]=df["Valore_Attuale_Calc"]-df["Valore_Carico_Calc"]
    df["Plusvalenza_Percentuale_Calc"]=np.where(df["Valore_Carico_Calc"]!=0,df["Plusvalenza_Calc"]/df["Valore_Carico_Calc"]*100,0)
    coupon=_num(df,"Cedola_Netta_Annua")
    df["Flusso_Cedolare_Annuo"]=coupon
    df["Yield_On_Cost"]=np.where(df["Valore_Carico_Calc"]!=0,coupon/df["Valore_Carico_Calc"]*100,0)
    dates=pd.to_datetime(df.get("Data_Scadenza"),errors="coerce")
    df["Giorni_Scadenza"]=(dates-pd.Timestamp.today().normalize()).dt.days
    y=df["Yield_On_Cost"]
    p=df["Plusvalenza_Percentuale_Calc"]
    d=df["Giorni_Scadenza"]

    # Scomposizione dello score di uscita
    df["Score_Plusvalenza"] = np.where(
        p > 20,
        50,
        np.where(p > 10, 30, np.where(p > 5, 10, 0))
    )

    df["Score_Scadenza"] = np.where(
        d < 730,
        20,
        np.where(d < 1825, 10, 0)
    )

    df["Score_Yield"] = np.where(
        y < 3,
        20,
        np.where(y < 5, 10, 0)
    )

    df["Score_Uscita"] = (
        df["Score_Plusvalenza"]
        + df["Score_Scadenza"]
        + df["Score_Yield"]
    )

    df["Classificazione"] = np.where(
        df["Score_Uscita"] > 40,
        "CANDIDATO ROTAZIONE",
        np.where(
            df["Score_Uscita"] >= 21,
            "MONITORARE",
            "MANTENERE"
        )
    )
    return df

def _barrier_distance(price, barrier):
    try:
        price=float(price); barrier=float(barrier)
        return (price-barrier)/price*100 if price else np.nan
    except: return np.nan

def enrich_certificates(certs, underlyings):
    c=certs.copy()
    if c.empty:
        return c
    u=underlyings.copy()
    if u.empty:
        c["Worst_Of"]=""
        c["Worst_Performance"]=np.nan
        c["Distanza_Barriera_Cedolare"]=np.nan
        c["Distanza_Barriera_Capitale"]=np.nan
        c["Classificazione"]="DATI INSUFFICIENTI"
        return c

    # Barriers belong to the underlying, not to the certificate price.
    u["Strike_num"]=pd.to_numeric(u.get("Strike"),errors="coerce")
    u["Price_num"]=pd.to_numeric(u.get("Prezzo_Attuale"),errors="coerce")
    u["Barrier_num"]=pd.to_numeric(u.get("Barriera"),errors="coerce")
    u["Performance"]=(u["Price_num"]-u["Strike_num"])/u["Strike_num"]*100
    u["Barrier_Distance"]=(u["Price_num"]-u["Barrier_num"])/u["Price_num"]*100

    rows=[]
    for isin,g in u.groupby("ISIN_CERTIFICATE"):
        valid=g.dropna(subset=["Performance"])
        if valid.empty:
            continue
        worst=valid.loc[valid["Performance"].idxmin()]
        rows.append({
            "ISIN":isin,
            "Worst_Of":worst.get("Sottostante",""),
            "Worst_Performance":worst["Performance"],
            "Worst_Distanza":worst["Barrier_Distance"],
            "Worst_Barrier":worst["Barrier_num"],
        })
    if rows:
        c=c.merge(pd.DataFrame(rows),left_on="ISIN",right_on="ISIN",how="left")
        c["Distanza_Barriera_Cedolare"]=c["Worst_Distanza"]
        c["Distanza_Barriera_Capitale"]=c["Worst_Distanza"]
        dist=c["Worst_Distanza"]
        c["Classificazione"]=np.select(
            [dist.notna() & (dist<15), dist.notna() & (dist<=30)],
            ["CRITICO","MONITORARE"],
            default="SICURO"
        )
    else:
        c["Worst_Of"]=""
        c["Distanza_Barriera_Cedolare"]=np.nan
        c["Distanza_Barriera_Capitale"]=np.nan
        c["Classificazione"]="DATI INSUFFICIENTI"
    return c

def portfolio_kpis(bonds, certs, liquidity):
    # =========================
    # VALORE ATTUALE
    # =========================

    bv = pd.to_numeric(
        bonds.get("Valore_Attuale_Calc", pd.Series(dtype=float)),
        errors="coerce"
    ).fillna(0).sum()

    cv = (
        pd.to_numeric(
            certs.get("Quantita", pd.Series(dtype=float)),
            errors="coerce"
        ).fillna(0)
        *
        pd.to_numeric(
            certs.get("Prezzo_Attuale", pd.Series(dtype=float)),
            errors="coerce"
        ).fillna(0)
    ).sum()

    lv = pd.to_numeric(
        liquidity.get("Saldo", pd.Series(dtype=float)),
        errors="coerce"
    ).fillna(0).sum()

    patrimonio = bv + cv + lv

    # =========================
    # CAPITALE INVESTITO
    # =========================

    bond_investito = pd.to_numeric(
        bonds.get("Valore_Carico_Calc", pd.Series(dtype=float)),
        errors="coerce"
    ).fillna(0).sum()

    cert_investito = (
        pd.to_numeric(
            certs.get("Quantita", pd.Series(dtype=float)),
            errors="coerce"
        ).fillna(0)
        *
        pd.to_numeric(
            certs.get("Prezzo_Carico", pd.Series(dtype=float)),
            errors="coerce"
        ).fillna(0)
    ).sum()

    capitale_investito = bond_investito + cert_investito

    # =========================
    # PLUS / MINUSVALENZE
    # =========================

    bond_pnl = pd.to_numeric(
        bonds.get("Plusvalenza_Calc", pd.Series(dtype=float)),
        errors="coerce"
    ).fillna(0).sum()

    cert_pnl = (
        pd.to_numeric(
            certs.get("Quantita", pd.Series(dtype=float)),
            errors="coerce"
        ).fillna(0)
        *
        (
            pd.to_numeric(
                certs.get("Prezzo_Attuale", pd.Series(dtype=float)),
                errors="coerce"
            ).fillna(0)
            -
            pd.to_numeric(
                certs.get("Prezzo_Carico", pd.Series(dtype=float)),
                errors="coerce"
            ).fillna(0)
        )
    ).sum()

    total_pnl = bond_pnl + cert_pnl

    # =========================
    # RENDIMENTI
    # =========================

    rendimento_totale_pct = (
        total_pnl / capitale_investito * 100
        if capitale_investito != 0
        else 0
    )

    # =========================
    # CEDOLE
    # =========================

    annual_coupon = pd.to_numeric(
        bonds.get("Flusso_Cedolare_Annuo", pd.Series(dtype=float)),
        errors="coerce"
    ).fillna(0).sum()

    rendimento_cedolare_patrimonio_pct = (
        annual_coupon / patrimonio * 100
        if patrimonio != 0
        else 0
    )

    rendimento_cedolare_capitale_pct = (
        annual_coupon / capitale_investito * 100
        if capitale_investito != 0
        else 0
    )

    return {
        # Valori attuali
        "bond_value": bv,
        "cert_value": cv,
        "liquidity": lv,
        "patrimonio": patrimonio,

        # Capitale investito
        "bond_investito": bond_investito,
        "cert_investito": cert_investito,
        "capitale_investito": capitale_investito,

        # Plus/minus
        "bond_pnl": bond_pnl,
        "cert_pnl": cert_pnl,
        "unrealized_pnl": total_pnl,
        "rendimento_totale_pct": rendimento_totale_pct,

        # Cedole
        "annual_coupon": annual_coupon,
        "rendimento_cedolare_patrimonio_pct":
            rendimento_cedolare_patrimonio_pct,
        "rendimento_cedolare_capitale_pct":
            rendimento_cedolare_capitale_pct,

        # Compatibilità con V1/V2
        "coupons_received": 0.0
    }
def coupon_forecast(bonds, coupons):
    c=coupons.copy()
    if not c.empty:
        c["Data"]=pd.to_datetime(c["Data_Cedola"],errors="coerce")
        c["Importo_Netto_Stimato"]=pd.to_numeric(c["Importo_Netto_Stimato"],errors="coerce").fillna(0)
        return c[["Data","ISIN","Descrizione","Importo_Netto_Stimato","Stato"]].sort_values("Data")
    if bonds.empty: return pd.DataFrame()
    rows=[]
    for _,r in bonds.iterrows():
        d=pd.to_datetime(r.get("Data_Prossima_Cedola"),errors="coerce")
        if pd.notna(d):
            rows.append({"Data":d,"ISIN":r.get("ISIN"),"Descrizione":r.get("Descrizione"),
                         "Importo_Netto_Stimato":float(r.get("Cedola_Netta_Annua",0))/2,
                         "Stato":"PREVISTA"})
    return pd.DataFrame(rows).sort_values("Data") if rows else pd.DataFrame()

def bond_rotation_candidates(bonds, catalog=None):
    if bonds.empty:
        return []

    out = []

    if catalog is None or catalog.empty:
        return []

    catalog = catalog.copy()

    # Normalizzazione dei campi numerici del catalogo
    for col in ["Prezzo", "Yield", "Cedola", "Duration"]:
        if col in catalog.columns:
            catalog[col] = pd.to_numeric(
                catalog[col],
                errors="coerce"
            )

    if "Scadenza" in catalog.columns:
        catalog["Scadenza"] = pd.to_datetime(
            catalog["Scadenza"],
            errors="coerce"
        )

    for _, r in bonds.iterrows():

        score = float(r.get("Score_Uscita", 0))

        if score <= 20:
            continue

        current_isin = str(r.get("ISIN", ""))

        # Recupera il rendimento a scadenza (YTM)
        # del Bond attualmente posseduto dal catalogo
        current_yield = 0.0

        if "Yield" in catalog.columns:
            current_match = catalog[
                catalog["ISIN"].astype(str) == current_isin
            ]

            if not current_match.empty:
                current_yield = float(
                    current_match.iloc[0]["Yield"]
                )
        current_duration = float(
            r.get("Duration", np.nan)
        )

        current_maturity = pd.to_datetime(
            r.get("Data_Scadenza"),
            errors="coerce"
        )

        # Esclude il Bond attualmente posseduto
        candidates = catalog[
            catalog["ISIN"].astype(str) != current_isin
        ].copy()
        # Il sostituto non deve avere un rating inferiore
        current_rating = ""

        if "Rating" in catalog.columns:
            current_match = catalog[
                catalog["ISIN"].astype(str) == current_isin
            ]

            if not current_match.empty:
                current_rating = str(
                    current_match.iloc[0]["Rating"]
                ).strip()

        if current_rating in {
            "AAA", "AA+", "AA", "AA-",
            "A+", "A", "A-",
            "BBB+", "BBB", "BBB-",
            "BB+", "BB", "BB-",
            "B+", "B", "B-"
        }:

            rating_order = {
                "AAA": 1,
                "AA+": 2,
                "AA": 3,
                "AA-": 4,
                "A+": 5,
                "A": 6,
                "A-": 7,
                "BBB+": 8,
                "BBB": 9,
                "BBB-": 10,
                "BB+": 11,
                "BB": 12,
                "BB-": 13,
                "B+": 14,
                "B": 15,
                "B-": 16
            }

            candidates["Rating_Score"] = (
                candidates["Rating"]
                .astype(str)
                .str.strip()
                .map(rating_order)
            )

            candidates = candidates[
                candidates["Rating_Score"].notna()
                & (
                    candidates["Rating_Score"]
                    <= rating_order[current_rating]
                )
            ]
        
        # Elimina candidati senza dati fondamentali
        required_cols = ["Yield", "Duration", "Scadenza"]

        available_required = [
            c for c in required_cols
            if c in candidates.columns
        ]

        if available_required:
            candidates = candidates.dropna(
                subset=available_required
            )

        if candidates.empty:
            out.append({
                "ISIN": current_isin,
                "Descrizione": r.get("Descrizione"),
                "Plusvalenza %": round(
                    float(r.get("Plusvalenza_Percentuale_Calc", 0)),
                    2
                ),
                "Yield On Cost %": round(
                    current_yield,
                    2
                ),
                "Giorni Scadenza": int(
                    r.get("Giorni_Scadenza", 0)
                ) if pd.notna(r.get("Giorni_Scadenza")) else None,
                "score": int(score),
                "Classificazione": r.get("Classificazione"),
                "Bond Sostitutivo": None,
                "Motivo": "Nessun candidato con Yield superiore"
            })
            continue

        # Differenza temporale dalla scadenza attuale
        if pd.notna(current_maturity):
            candidates["Diff_Scadenza_Anni"] = (
                (
                    candidates["Scadenza"]
                    - current_maturity
                ).abs()
                .dt.days
                / 365.25
            )
        else:
            candidates["Diff_Scadenza_Anni"] = 999

        # Differenza di Duration
        if pd.notna(current_duration):
            candidates["Diff_Duration"] = (
                candidates["Duration"]
                - current_duration
            ).abs()
        else:
            candidates["Diff_Duration"] = 999

        # Miglioramento del rendimento
        candidates["Delta_Yield"] = (
            candidates["Yield"]
            - current_yield
        )

        # Scomposizione dello score del candidato
        candidates["Score_Yield_Candidato"] = (
            candidates["Delta_Yield"] * 10
        )

        candidates["Penalita_Scadenza"] = (
            candidates["Diff_Scadenza_Anni"]
        )

        candidates["Penalita_Duration"] = (
            candidates["Diff_Duration"] * 2
        )

        candidates["Score_Candidato"] = (
            candidates["Score_Yield_Candidato"]
            - candidates["Penalita_Scadenza"]
            - candidates["Penalita_Duration"]
        )

        best = candidates.sort_values(
            "Score_Candidato",
            ascending=False
        ).iloc[0]

        out.append({
            "ISIN": current_isin,
            "Descrizione": r.get("Descrizione"),
            "Plusvalenza %": round(
                float(r.get("Plusvalenza_Percentuale_Calc", 0)),
                2
            ),
            "Yield On Cost %": round(
                current_yield,
                2
            ),
            "Giorni Scadenza": int(
                r.get("Giorni_Scadenza", 0)
            ) if pd.notna(r.get("Giorni_Scadenza")) else None,
            "score": int(score),
            "Classificazione": r.get("Classificazione"),
            "Bond Sostitutivo": best.get("ISIN"),
            "Descrizione Sostitutivo": best.get("Descrizione"),
            "Yield Sostitutivo": round(
                float(best.get("Yield")),
                2
            ),
            "Duration Sostitutivo": round(
                float(best.get("Duration")),
                2
            ),
            "Scadenza Sostitutivo": best.get(
                "Scadenza"
            ).strftime("%Y-%m-%d"),

            "Delta Yield": round(
                float(best.get("Delta_Yield")),
                2
            ),
            "Score Yield Candidato": round(
                float(best.get("Score_Yield_Candidato")),
                2
            ),
            "Penalita Scadenza": round(
                float(best.get("Penalita_Scadenza")),
                2
            ),
            "Penalita Duration": round(
                float(best.get("Penalita_Duration")),
                2
            ),
            "Score_Candidato": round(
                float(best.get("Score_Candidato")),
                2
            )

        })

    return sorted(
        out,
        key=lambda x: x["score"],
        reverse=True
    )


def certificate_alerts(certs):
    if certs.empty: return []
    return [f"{r.get('ISIN')} — {r.get('Worst_Of','')} — distanza barriera {r.get('Distanza_Barriera_Capitale',np.nan):.1f}%"
            for _,r in certs.iterrows() if r.get("Classificazione")=="CRITICO"]
