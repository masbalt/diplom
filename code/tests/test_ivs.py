"""Юнит-тесты на синтетических данных: python -m unittest discover -s code/tests"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivs import arbitrage, black76, experiment, forecast, gp, neural, pilot, quotes, recon, ssvi  # noqa: E402
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


SERIES = pd.DataFrame({"expiry": pd.to_datetime(["2026-10-15", "2026-11-19", "2026-12-17"]),
                       "T": [0.04, 0.14, 0.22], "atm": [0.16, 0.17, 0.18]})


def ssvi_surface() -> ssvi.Surface:
    T = SERIES["T"].to_numpy()
    return ssvi.Surface(T, SERIES["atm"].to_numpy() ** 2 * T, rho=0.4, eta=1.0, gamma=0.4)


class SsviTest(unittest.TestCase):
    def test_fit_recovers_parameters(self):
        true = ssvi_surface()
        pts = pd.DataFrame([(T, x * np.sqrt(T)) for T in true.T for x in np.linspace(-0.4, 0.4, 9)], columns=["T", "k"])
        pts["iv"] = true.iv(pts["k"], pts["T"])
        fit = ssvi.fit(pts["k"], pts["T"], pts["iv"], ridge=0.0, f_scale=None)
        self.assertAlmostEqual(fit.rho, 0.4, places=3)
        np.testing.assert_allclose(fit.iv(pts["k"], pts["T"]), pts["iv"], atol=1e-6)

    def test_no_arbitrage_by_construction(self):
        # Параметры на границе допустимой области и серия без точек (theta интерполируется).
        s = ssvi.Surface(np.array([0.04, 0.22]), np.array([0.3, 0.3]) ** 2 * np.array([0.04, 0.22]), 0.95, 2 / 1.95, 0.5)
        grid = arbitrage.make_grid(SERIES.assign(atm=0.3), z_max=3.0, step=0.005)
        v = arbitrage.violations(grid, s.iv(grid["k"], grid["T"]), tol=1e-9)
        self.assertFalse(v["butterfly_any"] or v["calendar_any"])


class ArbitrageTest(unittest.TestCase):
    def setUp(self):
        self.grid = arbitrage.make_grid(SERIES, z_max=2.0, step=0.005)
        self.iv = ssvi_surface().iv(self.grid["k"], self.grid["T"])

    def test_grid_is_nested(self):
        nodes = [set(g["j"]) for _, g in self.grid.groupby("T", sort=True)]
        self.assertTrue(nodes[0] <= nodes[1] <= nodes[2])

    def test_repair_removes_calendar_violation(self):
        iv = self.iv.copy()
        iv[self.grid["T"] == 0.22] *= 0.6          # дальняя серия дешевле ближней
        self.assertTrue(arbitrage.violations(self.grid, iv, tol=1e-6)["calendar_any"])
        fixed = arbitrage.repair(self.grid, iv)
        v = arbitrage.violations(self.grid, fixed, tol=1e-6)
        self.assertFalse(v["butterfly_any"] or v["calendar_any"])

    def test_repair_removes_butterfly_violation_locally(self):
        iv = self.iv.copy()
        j = self.grid.index[self.grid["T"] == 0.14][10]
        iv[j] -= 0.03                                # провал в одном узле
        self.assertTrue(arbitrage.violations(self.grid, iv, tol=1e-6)["butterfly_any"])
        fixed = arbitrage.repair(self.grid, iv)
        self.assertFalse(arbitrage.violations(self.grid, fixed, tol=1e-6)["butterfly_any"])
        self.assertLess((np.abs(fixed - iv) > 1e-6).sum(), 6)   # исправление затронуло несколько узлов

    def test_weights_choose_which_series_moves(self):
        iv = self.iv.copy()
        far = (self.grid["T"] == 0.22).to_numpy()
        iv[far] *= 0.6
        w = np.where(far, 1.0, 100.0)                # ближним сериям метод доверяет
        fixed = arbitrage.repair(self.grid, iv, w)
        self.assertLess(np.abs(fixed - iv)[~far].max(), 1e-6)
        self.assertGreater(np.abs(fixed - iv)[far].max(), 0.01)

    def test_arbitrage_free_surface_is_unchanged(self):
        np.testing.assert_array_equal(arbitrage.repair(self.grid, self.iv), self.iv)


class GpTest(unittest.TestCase):
    P = {**gp.INIT}

    def day(self, shift: float = 0.01) -> pd.DataFrame:
        return synthetic_day(level_shift=shift).assign(n_candles=10.0)

    def test_recovers_level_shift(self):
        d = self.day()
        vis, hid = d.iloc[::2], d.iloc[1::2].reset_index(drop=True)
        pred, sd = gp.predict(vis, hid, self.P, return_sd=True)
        self.assertLess(np.abs(pred - hid["iv"]).max(), 0.004)       # вчерашняя кривая ошибается на 0.01
        self.assertTrue((sd > 0).all())

    def test_no_visible_points_in_series_gives_larger_uncertainty(self):
        d = self.day()
        first = d["expiry"] == d["expiry"].iloc[0]
        _, sd_seen = gp.predict(d[first], d[first].reset_index(drop=True), self.P, return_sd=True)
        _, sd_unseen = gp.predict(d[first], d[~first].reset_index(drop=True), self.P, return_sd=True)
        self.assertGreater(sd_unseen.mean(), sd_seen.mean())

    def test_fit_improves_likelihood(self):
        days = [gp.arrays(synthetic_day(0.01 * (-1) ** i, seed=i).assign(n_candles=10.0)) for i in range(6)]
        fixed = {k: v for k, v in gp.INIT.items() if k not in ("g0", "a")}
        P = gp.fit(days, fixed=fixed)
        self.assertLess(P["nll"], gp.neg_log_lik(gp.INIT, days))
        self.assertGreater(P["g0"], 0.005)           # сдвиг уровня 1 п.п. больше начального масштаба


class MaskTest(unittest.TestCase):
    def test_far_and_topn(self):
        rng = np.random.default_rng(0)
        d = pd.concat([synthetic_day(), synthetic_day().assign(expiry=pd.Timestamp("2027-03-18"), T=0.4)],
                      ignore_index=True)
        d["date"], d["z"] = pd.Timestamp("2026-10-01"), d["x"] / 0.16
        d["n_candles"], d["vol_sync"] = rng.integers(1, 50, len(d)), rng.integers(1, 500, len(d))
        p = {"MASKS": [0.5], "SEEDS": 1, "WINGS_Z": 1.0, "FAR_KEEP": 2, "TOPN": [5]}
        configs = {c[0]: c[3] for c in experiment.mask_configs(d, p)}
        self.assertEqual(set(d["T"].iloc[configs["far"]]), {0.4})                 # скрыта только дальняя серия
        visible = d.drop(index=configs["topn"])
        self.assertEqual(len(visible), 5)
        self.assertGreaterEqual(visible["n_candles"].min(), d["n_candles"].iloc[configs["topn"]].max())


class ForecastTest(unittest.TestCase):
    E = pd.Timestamp("2026-10-15")

    def smiles(self, T: float = 0.10) -> recon.Smiles:
        k = np.linspace(-0.2, 0.2, 41)
        return recon.Smiles(pd.DataFrame({"expiry": self.E, "T": T, "k": k, "iv_settle": 0.16 + 2.0 * k**2}))

    def test_transfer_rules(self):
        sm, k, T = self.smiles(T=0.10), np.array([-0.05, 0.0, 0.08]), 0.05
        curve = lambda q: 0.16 + 2.0 * q**2   # noqa: E731
        np.testing.assert_allclose(forecast.transfer(sm, self.E, k, T, "k"), curve(k), atol=1e-4)
        # Та же x = k / sqrt(T): срок уменьшился вдвое, значит кривая читается в точке k * sqrt(2).
        np.testing.assert_allclose(forecast.transfer(sm, self.E, k, T, "x"), curve(k * np.sqrt(2)), atol=1e-4)
        # Тот же страйк: фьючерс вырос на 2 %, денежность страйка в день t была больше на ln(1.02).
        strike = 84000.0 * np.exp(k)
        np.testing.assert_allclose(forecast.transfer(sm, self.E, k, T, "strike", strike, 84000.0 / 1.02),
                                   curve(k + np.log(1.02)), atol=1e-4)

    def test_carried_correction(self):
        d = synthetic_day(level_shift=0.01).assign(n_candles=10.0)
        vis = d.assign(today=d["hist"])                       # сделки дня t на 1 п.п. выше кривой дня t
        np.testing.assert_allclose(forecast.smooth_resid(vis, d, gp.INIT), 0.01, atol=0.004)
        sm = recon.Smiles(d.assign(iv_settle=d["hist"]))
        hist_x, dyn = forecast.dyn_prior(d, sm, vis, gp.INIT, 0.5)
        np.testing.assert_allclose(hist_x, d["hist"], atol=1e-6)   # срок не менялся: перенос в x совпадает с кривой
        np.testing.assert_allclose(dyn - hist_x, 0.005, atol=0.002)
        np.testing.assert_allclose(forecast.dyn_prior(d, sm, None, gp.INIT, 0.5)[1], hist_x)   # без сделок — сама кривая

    def test_ar_beta_recovers_mean_reversion(self):
        rng = np.random.default_rng(0)
        f = np.zeros((4000, 2))
        for i in range(1, len(f)):
            f[i] = 0.8 * f[i - 1] + rng.normal(size=2)
        fac = pd.DataFrame(f, index=pd.date_range("2000-01-03", periods=len(f), freq="B"))
        beta = forecast.ar_beta(fac, fac, 1, fac.index[-1])
        np.testing.assert_allclose(beta, -0.2, atol=0.03)     # за день фактор теряет 20 % отклонения


class NeuralTest(unittest.TestCase):
    P = {"NP_EPOCHS": 25, "NP_BATCH": 4, "NP_LR": 3e-3, "NP_HUBER": 2.0, "NP_SEEDS": 1,
         "WINGS_Z": 1.0, "FAR_KEEP": 1, "TOPN": [5, 10]}

    def day(self, i: int) -> pd.DataFrame:
        d = synthetic_day(level_shift=0.02 * (-1) ** i, seed=i)
        return d.assign(date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), z=d["x"] / 0.16,
                        n_candles=10.0, vol_sync=np.arange(len(d), dtype=float))

    def test_learns_level_shift(self):
        train = pd.concat([self.day(i) for i in range(12)], ignore_index=True)
        models = neural.fit_all(train, {"np": "hist", "np_cb": None}, self.P, workers=2)
        d = self.day(20)
        vis, hid = d.iloc[::2], d.iloc[1::2].reset_index(drop=True)
        for name in models:
            pred = neural.predict(vis, hid, models[name], name)
            self.assertEqual(pred.shape, (len(hid),))
            # Вчерашняя кривая ошибается на 2 п.п.; сеть по видимым сделкам снимает большую часть сдвига.
            self.assertLess(np.abs(pred - hid["iv"]).mean(), 0.01)


class SummaryTest(unittest.TestCase):
    def test_worst_prior_pair(self):
        near, far = pd.Timestamp("2026-11-19"), pd.Timestamp("2027-01-21")
        d = pd.DataFrame({"date": pd.Timestamp("2026-11-11"), "expiry": [near, near, far, far],
                          "strike": [84000.0, 85000.0, 78000.0, 90000.0], "iv": 0.20,
                          "hist": [0.21, 0.19, 0.50, 0.20]})
        res = pd.concat([d, d], ignore_index=True)   # сделка, скрытая в двух схемах, считается один раз
        worst = experiment.worst_prior_pair(res)
        self.assertEqual((worst["date"], worst["expiry"]), ("2026-11-11", "2027-01-21"))
        self.assertEqual((worst["trades"], worst["pairs"]), (2, 2))
        self.assertAlmostEqual(worst["hist_err_vp"], 100 * np.sqrt(0.30**2 / 2))
        self.assertGreater(worst["share_sse"], 0.99)       # почти вся ошибка вчерашней кривой — в одной паре


class QuotesTest(unittest.TestCase):
    def test_summary(self):
        q = pd.DataFrame({"expiry": pd.Timestamp("2026-10-15"), "bid": [0.15, 0.15, 0.10], "ask": [0.17, 0.17, 0.20]})
        for m in quotes.METHODS:
            q[m] = [0.16, 0.18, 0.12]            # внутри спреда первая и третья заявки
        s = quotes.summary(q, max_spread=0.03).set_index(["subset", "method"])
        self.assertAlmostEqual(s.loc[("all", "gp_dyn"), "inside"], 2 / 3)
        self.assertEqual(s.loc[("narrow", "gp_dyn"), "n"], 2)          # третья заявка шире 3 п.п.
        self.assertAlmostEqual(s.loc[("narrow", "gp_dyn"), "inside"], 0.5)
        self.assertAlmostEqual(s.loc[("narrow", "gp_dyn"), "mad_vp"], 1.0)


if __name__ == "__main__":
    unittest.main()
