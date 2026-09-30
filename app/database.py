
from pathlib import Path
import sqlite3
import pandas as pd

TABLES = {
    "BOND": "bond",
    "CERTIFICATES": "certificates",
    "SOTTOSTANTI": "sottostanti",
    "CEDOLE": "cedole",
    "PROPOSTE_ROTATORIE": "proposte_rotatorie",
}

def _read_csv(path, columns=None):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])
    except Exception as exc:
        raise RuntimeError(f"Errore nella lettura di {path.name}: {exc}") from exc

def load_portfolio_data(data_dir, root=None):
    data_dir = Path(data_dir)
    root = Path(root) if root else data_dir
    return (
        _read_csv(data_dir/"BOND_PORTAFOGLIO.csv"),
        _read_csv(data_dir/"CERTIFICATES_PORTAFOGLIO.csv"),
        _read_csv(data_dir/"SOTTOSTANTI_CERTIFICATES.csv"),
        _read_csv(data_dir/"CALENDARIO_CEDOLE.csv"),
        _read_csv(data_dir/"CATALOGO_BOND.csv"),
        _read_csv(root/"LIQUIDITA.csv"),
        _read_csv(data_dir/"TITOLARI.csv"),
    )

def sync_to_sqlite(root, bonds, certs, underlyings, coupons, proposals=None):
    """Create the V1 operational SQLite database from the current CSV data."""
    db_dir = Path(root) / "database"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "portfolio.db"
    conn = sqlite3.connect(db_path)
    datasets = {
        "bond": bonds,
        "certificates": certs,
        "sottostanti": underlyings,
        "cedole": coupons,
        "proposte_rotatorie": proposals if proposals is not None else pd.DataFrame(),
    }
    for table, df in datasets.items():
        if df.empty:
            # Keep the CSV schema even when the CSV currently contains only headers.
            cols = list(df.columns)
            if not cols:
                cols = ["id"]
            quoted = ", ".join(f'"{c}" TEXT' for c in cols)
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.execute(f'CREATE TABLE "{table}" ({quoted})')
        else:
            df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    return db_path
