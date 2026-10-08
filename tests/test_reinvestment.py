
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from reinvestment import allocate_lots


class TestReinvestment(unittest.TestCase):

    def test_allocate_lots(self):
        candidates = pd.DataFrame([
            {
                "ISIN": "TEST_BTP_A",
                "Descrizione": "BTP A",
                "Prezzo": 100.0,
                "Score_Yield": 90,
            },
            {
                "ISIN": "TEST_BTP_B",
                "Descrizione": "BTP B",
                "Prezzo": 95.0,
                "Score_Yield": 80,
            },
            {
                "ISIN": "TEST_BTP_C",
                "Descrizione": "BTP C",
                "Prezzo": 105.0,
                "Score_Yield": 70,
            },
        ])

        capitale = 10000.0

        allocation, residual = allocate_lots(
            candidates,
            capitale,
            max_positions=3,
            profile="rendimento",
        )

        self.assertFalse(allocation.empty)
        self.assertLessEqual(len(allocation), 3)

        self.assertTrue(
            (allocation["Lotti_Allocati"] >= 1).all()
        )

        self.assertTrue(
            (
                allocation["Nominale_Allocato"]
                == allocation["Lotti_Allocati"] * 1000
            ).all()
        )

        invested = allocation["Capitale_Investito"].sum()

        self.assertLessEqual(invested, capitale)
        self.assertGreaterEqual(residual, 0)

        self.assertAlmostEqual(
            invested + residual,
            capitale,
            places=6,
        )

    def test_cash_flow_netto_btp(self):
        from reinvestment import calculate_net_cashflow

        risultato = calculate_net_cashflow(
            nominale=10000,
            cedola_percentuale=4.0,
            aliquota_fiscale=0.125,
        )

        self.assertAlmostEqual(
            risultato,
            350.0,
            places=2,
        )

    def test_candidates_cashflow_netto(self):
        from reinvestment import reinvestment_candidates

        catalog = pd.DataFrame([
            {
                "ISIN": "TEST_BTP_4",
                "Descrizione": "BTP Test 4%",
                "Emittente": "Italia",
                "Prezzo": 100.0,
                "Yield": 4.0,
                "Rating": "BBB",
                "Scadenza": "2035-01-01",
                "Cedola": 4.0,
                "Duration": 7.0,
            }
        ])

        candidates = reinvestment_candidates(
            catalog,
            amount=10000.0,
            bonds=None,
            profile="cash_flow",
        )

        self.assertFalse(candidates.empty)

        self.assertIn(
            "CashFlow_Netto_Annuale",
            candidates.columns,
        )

        row = candidates.iloc[0]

        self.assertAlmostEqual(
            row["CashFlow_Annuale"],
            400.0,
            places=2,
        )

        self.assertAlmostEqual(
            row["CashFlow_Netto_Annuale"],
            350.0,
            places=2,
        )

    def test_ladder_evits_excessive_concentration(self):
        candidates = pd.DataFrame([
            {
                "ISIN": "BTP_A",
                "Descrizione": "BTP A",
                "Prezzo": 100.0,
                "Score_Diversificazione": 90,
            },
            {
                "ISIN": "BTP_B",
                "Descrizione": "BTP B",
                "Prezzo": 100.0,
                "Score_Diversificazione": 80,
            },
            {
                "ISIN": "BTP_C",
                "Descrizione": "BTP C",
                "Prezzo": 100.0,
                "Score_Diversificazione": 70,
            },
        ])

        allocation, residual = allocate_lots(
            candidates,
            amount=10000.0,
            max_positions=3,
            profile="ladder",
        )

        self.assertEqual(len(allocation), 3)

        total = allocation["Capitale_Investito"].sum()

        self.assertAlmostEqual(
            total + residual,
            10000.0,
            places=6,
        )

        self.assertLessEqual(
            allocation["Capitale_Investito"].max() / total,
            0.50,
        )
    def test_ladder_prezzi_differenti(self):
        candidates = pd.DataFrame([
            {
                "ISIN": "BTP_75",
                "Descrizione": "BTP prezzo 75",
                "Prezzo": 75.0,
                "Score_Diversificazione": 90,
            },
            {
                "ISIN": "BTP_95",
                "Descrizione": "BTP prezzo 95",
                "Prezzo": 95.0,
                "Score_Diversificazione": 80,
            },
            {
                "ISIN": "BTP_110",
                "Descrizione": "BTP prezzo 110",
                "Prezzo": 110.0,
                "Score_Diversificazione": 70,
            },
        ])

        capitale = 10000.0

        allocation, residual = allocate_lots(
            candidates,
            amount=capitale,
            max_positions=3,
            profile="ladder",
        )

        self.assertEqual(len(allocation), 3)

        invested = allocation["Capitale_Investito"].sum()

        self.assertLessEqual(invested, capitale)
        self.assertGreaterEqual(residual, 0)

        self.assertAlmostEqual(
            invested + residual,
            capitale,
            places=6,
        )

        self.assertTrue(
            (
                allocation["Nominale_Allocato"]
                == allocation["Lotti_Allocati"] * 1000
            ).all()
        )

        self.assertLessEqual(
            allocation["Capitale_Investito"].max() / invested,
            0.50,
        )

    def test_limite_concentrazione_profili(self):
        candidates = pd.DataFrame([
            {
                "ISIN": "BTP_A",
                "Descrizione": "BTP A",
                "Prezzo": 100.0,
                "Score_Yield": 90,
                "Score_CashFlow": 90,
            },
            {
                "ISIN": "BTP_B",
                "Descrizione": "BTP B",
                "Prezzo": 100.0,
                "Score_Yield": 80,
                "Score_CashFlow": 80,
            },
            {
                "ISIN": "BTP_C",
                "Descrizione": "BTP C",
                "Prezzo": 100.0,
                "Score_Yield": 70,
                "Score_CashFlow": 70,
            },
        ])

        for profile in ["cash_flow", "rendimento"]:
            with self.subTest(profile=profile):
                allocation, residual = allocate_lots(
                    candidates,
                    amount=10000.0,
                    max_positions=3,
                    profile=profile,
                )

                invested = allocation["Capitale_Investito"].sum()

                self.assertGreater(invested, 0)

                self.assertLessEqual(
                    allocation["Capitale_Investito"].max()
                    / invested,
                    0.50,
                )

                self.assertAlmostEqual(
                    invested + residual,
                    10000.0,
                    places=6,
                )


    def test_concentrazione_prezzi_reali(self):
        candidates = pd.DataFrame([
            {
                "ISIN": "BTP_A",
                "Descrizione": "BTP A",
                "Prezzo": 98.34,
                "Score_CashFlow": 90,
                "Score_Yield": 90,
            },
            {
                "ISIN": "BTP_B",
                "Descrizione": "BTP B",
                "Prezzo": 91.78,
                "Score_CashFlow": 80,
                "Score_Yield": 80,
            },
            {
                "ISIN": "BTP_C",
                "Descrizione": "BTP C",
                "Prezzo": 95.06,
                "Score_CashFlow": 70,
                "Score_Yield": 70,
            },
        ])

        for profile in ["cash_flow", "rendimento"]:
            with self.subTest(profile=profile):
                allocation, residual = allocate_lots(
                    candidates,
                    amount=10000.0,
                    max_positions=3,
                    profile=profile,
                )

                invested = allocation["Capitale_Investito"].sum()

                self.assertGreater(invested, 0)

                self.assertLessEqual(
                    allocation["Capitale_Investito"].max()
                    / invested,
                    0.50 + 1e-9,
                )

                self.assertAlmostEqual(
                    invested + residual,
                    10000.0,
                    places=6,
                )


    def test_riutilizzo_liquidita_residua(self):
        candidates = pd.DataFrame([
            {
                "ISIN": "BTP_A",
                "Descrizione": "BTP A",
                "Prezzo": 98.34,
                "Score_CashFlow": 90,
            },
            {
                "ISIN": "BTP_B",
                "Descrizione": "BTP B",
                "Prezzo": 91.78,
                "Score_CashFlow": 80,
            },
            {
                "ISIN": "BTP_C",
                "Descrizione": "BTP C",
                "Prezzo": 95.06,
                "Score_CashFlow": 70,
            },
        ])

        allocation, residual = allocate_lots(
            candidates,
            amount=10000.0,
            max_positions=3,
            profile="cash_flow",
        )

        invested = allocation["Capitale_Investito"].sum()

        self.assertLessEqual(
            allocation["Capitale_Investito"].max() / invested,
            0.50 + 1e-9,
        )

        self.assertAlmostEqual(
            invested + residual,
            10000.0,
            places=6,
        )

        # Con questi prezzi è possibile investire almeno
        # 9.506 euro senza superare il 50% per posizione.
        self.assertGreaterEqual(
            invested,
            9506.0 - 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
