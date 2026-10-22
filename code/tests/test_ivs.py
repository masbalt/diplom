"""Юнит-тесты на синтетических данных: python -m unittest discover -s code/tests"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivs import black76, pilot, recon  # noqa: E402
from ivs.contracts import parse_secids  # noqa: E402


class Black76Test(unittest.TestCase):
    def test_roundtrip(self):
        F = np.full(4, 84000.0)
        K = np.array([75000.0, 82000.0, 84000.0, 92000.0])
        T = np.array([0.02, 0.1, 0.25, 0.5])
        sigma = np.array([0.30, 0.18, 0.16, 0.22])
        is_call = K >= F
        p = black76.price(F, K, T, sigma, is_call)
        np.testing.assert_allclose(black76.implied_vol(p, F, K, T, is_call), sigma, atol=1e-8)

    def test_put_call_parity_undiscounted(self):
        c = black76.price(84000.0, 80000.0, 0.1, 0.2, True)
        p = black76.price(84000.0, 80000.0, 0.1, 0.2, False)
        self.assertAlmostEqual(float(c - p), 4000.0, places=6)

    def test_price_outside_bounds_is_nan(self):
        # Цена колла ниже внутренней стоимости и выше F — IV не существует.
        iv = black76.implied_vol([3000.0, 90000.0], 84000.0, 80000.0, 0.1, True)
        self.assertTrue(np.isnan(iv).all())


class SecidTest(unittest.TestCase):
    def test_parse(self):
        p = parse_secids(pd.Series(["Si80000BL6", "Si80000BX6", "RI107500BO0D", "Si61250BD6_2016", "CR11.8BJ6B",
                                    "garbage"]))
        self.assertEqual(list(p["opt_type"][:5]), ["C", "P", "P", "C", "C"])
        self.assertEqual(list(p["month"][:5]), [12, 12, 3, 4, 10])
        self.assertEqual(p["strike"].iloc[4], 11.8)
        # Колл и пут одной экспирации попадают в одну группу; неделя и год различают группы.
        self.assertEqual(p["group"].iloc[0], p["group"].iloc[1])
        self.assertTrue(p["group"].iloc[2].endswith("|D|"))
        self.assertTrue(p["group"].iloc[3].endswith("|2016"))
        self.assertTrue(pd.isna(p["group"].iloc[5]))


def synthetic_day(level_shift: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for e, T in [(pd.Timestamp("2026-10-15"), 0.04), (pd.Timestamp("2026-11-19"), 0.14)]:
        k = np.sort(rng.uniform(-0.08, 0.08, 12))
        smile = 0.16 + 0.8 * k**2 / np.sqrt(T) - 0.1 * k
        rows.append(pd.DataFrame({"expiry": e, "T": T, "k": k, "iv": smile + level_shift, "hist": smile}))
    d = pd.concat(rows, ignore_index=True)
    d["x"], d["sqrtT"] = d["k"] / np.sqrt(d["T"]), np.sqrt(d["T"])
    return d


class ReconTest(unittest.TestCase):
    def test_hist_cb_recovers_level_shift(self):
        d = synthetic_day(level_shift=0.01)
        vis, hid = d.iloc[::2], d.iloc[1::2].reset_index(drop=True)
        err = recon.hist_cb(vis, hid) - hid["iv"]
        self.assertLess(np.abs(err).max(), 0.002)  # сдвиг 1 п.п. снят почти полностью

    def test_cb_quad_series_without_points(self):
        d = synthetic_day()
        vis = d[d["expiry"] == d["expiry"].iloc[0]]
        hid = d[d["expiry"] != d["expiry"].iloc[0]].reset_index(drop=True)
        pred = recon.cb_quad(vis, hid)
        self.assertEqual(pred.shape, (len(hid),))
        self.assertTrue(np.isfinite(pred).all())


class PilotStatsTest(unittest.TestCase):
    def test_rmse_ci_contains_point(self):
        rng = np.random.default_rng(1)
        r = pd.DataFrame({"date": np.repeat(pd.date_range("2026-01-01", periods=30), 10),
                          "iv": 0.2, "a": 0.2 + rng.normal(0, 0.01, 300), "b": 0.2 + rng.normal(0, 0.02, 300)})
        t = pilot.rmse_with_ci(r, ["a", "b"], n_boot=200).set_index("method")
        self.assertTrue(((t["lo"] <= t["rmse"]) & (t["rmse"] <= t["hi"])).all())
        _, _, hi = pilot.diff_with_ci(r, "a", "b", n_boot=200)
        self.assertLess(hi, 0)  # a заведомо точнее b

    def test_tex_number_format(self):
        self.assertEqual(pilot._num(3.776), "3{,}78")
        self.assertEqual(pilot._num(21217, 0), "21\\,217")


if __name__ == "__main__":
    unittest.main()
