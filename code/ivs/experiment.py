"""Основной эксперимент: восстановление скрытых сделок при разной разреженности доски.

Отличия от пилота (ivs.pilot):
  * хронологическое разбиение: параметры предлагаемого метода оцениваются на TRAIN_RANGE,
    все числа качества считаются на TEST_RANGE (строго позже);
  * добавлены SSVI и предлагаемый метод (ivs.gp) с упрощёнными вариантами;
  * новые схемы скрытия: ``far`` (видны две ближние серии) и ``topn`` (видны N самых
    торгуемых контрактов дня);
  * предлагаемый метод оценивается и с шагом исправления арбитража (``gp_af``);
  * отдельные исследования: устаревание истории, статический арбитраж, режимы рынка,
    точность оценки неопределённости.

Случайные маски совпадают с пилотом (тот же генератор и порядок), поэтому числа старых
методов воспроизводятся один в один.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

from . import arbitrage, gp, pilot, recon, ssvi

CURRENT_BOARD = ["cb_quad", "cb_svi", "cb_ssvi"]
BASE_METHODS = ["hist", "cb_quad", "cb_svi", "cb_ssvi", "hist_cb", "gp", "gp_af", "moex_today"]
# Варианты предлагаемого метода: что используется как априор и какие параметры выключены.
VARIANTS = {
    "gp": {"prior": "hist"},
    "gp_cb": {"prior": None, "init": gp.INIT_NO_PRIOR},                       # без истории
    "gp_indep": {"prior": "hist", "flags": {"indep": True},                   # без связей между сериями
                 "fixed": {"g0": gp.OFF, "g1": gp.OFF, "g2": gp.OFF, "ts": gp.OFF, "ell_T": 1.0}},
    "gp_homo": {"prior": "hist", "fixed": {"b": gp.OFF}},                     # одинаковый шум сделок
    "gp_param": {"prior": "hist", "fixed": {"tloc": gp.OFF, "ell_x": 0.1}},   # только уровень, наклон, кривизна
}
ARB_METHODS = ["cb_quad", "cb_svi", "cb_ssvi", "hist", "hist_cb", "gp", "moex_today"]
KEEP = ["date", "expiry", "strike", "T", "k", "x", "z", "iv", "n_candles"]
SD_FLOOR = 1e-4   # нижняя граница неопределённости метода в узле сетки, доли IV


# --- наблюдения и обучение ------------------------------------------------------

def load(data_dir: Path, asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    board = pd.read_csv(Path(data_dir) / "processed" / f"board_daily_{asset}.csv.gz",
                        parse_dates=["date", "expiry"], low_memory=False)
    sync = pd.read_csv(Path(data_dir) / "processed" / f"trades_sync_{asset}.csv.gz", parse_dates=["date", "expiry"])
    return board, sync


def observations(board: pd.DataFrame, sync: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Наблюдения за TRAIN_RANGE и TEST_RANGE с колонкой ``part`` (train / test)."""
    lo, hi = pd.Timestamp(p["TRAIN_RANGE"][0]), pd.Timestamp(p["TEST_RANGE"][1])
    sync = sync[(sync["date"] >= lo) & (sync["date"] <= hi)]
    obs = pilot.build_observations(board, sync, p, lags=tuple(h for h in p["LAGS"] if h != 1))
    obs["hist_lag1"] = obs["hist"]
    obs["part"] = np.where(obs["date"] >= pd.Timestamp(p["TEST_RANGE"][0]), "test", "train")
    obs = obs[(obs["part"] == "test") | (obs["date"] <= pd.Timestamp(p["TRAIN_RANGE"][1]))]
    return obs.reset_index(drop=True)


def _fit_variant(job: tuple) -> tuple[str, dict]:
    name, days, spec = job
    P = gp.fit(days, spec.get("init", gp.INIT), spec.get("fixed"), spec.get("flags"))
    return name, P


def train(obs: pd.DataFrame, p: dict, workers: int) -> dict:
    """Параметры всех вариантов метода по дням обучающего периода (в том числе для лагов истории)."""
    tr = obs[obs["part"] == "train"]
    jobs = []
    for name, spec in VARIANTS.items():
        days = [gp.arrays(g, spec["prior"]) for _, g in tr.groupby("date")]
        jobs.append((name, days, spec))
    for h in p["LAGS"]:
        col = f"hist_lag{h}"
        days = [gp.arrays(g, col) for _, g in tr.dropna(subset=[col]).groupby("date")]
        jobs.append((f"gp_lag{h}", days, {}))
    with ProcessPoolExecutor(workers) as pool:
        return dict(pool.map(_fit_variant, jobs))


# --- вчерашние кривые и сетка дня -----------------------------------------------

def day_context(board: pd.DataFrame) -> tuple[dict, dict, pd.Series]:
    """Расчётные кривые по дням, предыдущий торговый день и медианная цена фьючерса дня."""
    dates = sorted(board["date"].unique())
    return pilot.settle_smiles(board), dict(zip(dates[1:], dates[:-1])), board.groupby("date")["F"].median()


def day_grid(obs: pd.DataFrame, s_prev, p: dict, s_today=None) -> pd.DataFrame:
    """Сетка для проверки арбитража по сериям дня; в узлах — вчерашняя (и сегодняшняя) кривая."""
    ser = obs[["expiry", "T"]].drop_duplicates().copy()
    ser["atm"] = [float(s_prev(e, np.array([0.0]), T)[0]) for e, T in zip(ser["expiry"], ser["T"])]
    grid = arbitrage.make_grid(ser, p["ARB_Z_MAX"], p["ARB_STEP"])
    for col, smiles in (("hist", s_prev), ("today", s_today)):
        if smiles is not None:
            grid[col] = np.nan
            for e, g in grid.groupby("expiry"):
                grid.loc[g.index, col] = smiles(e, g["k"].to_numpy(), float(g["T"].iloc[0]))
    return grid


def gp_weight(vis: pd.DataFrame, grid: pd.DataFrame, P: dict) -> np.ndarray:
    """Вес узла для исправления арбитража: обратная неопределённость метода в узле."""
    return 1.0 / np.maximum(gp.predict(vis, grid, P, return_sd=True)[1], SD_FLOOR)


def repaired(method, vis: pd.DataFrame, hid: pd.DataFrame, grid: pd.DataFrame, p: dict, weight=None):
    """Прогноз метода в ``hid`` до и после исправления арбитража на сетке, IV в узлах и поправка."""
    iv = np.clip(method(vis, grid), p["ARB_IV_MIN"], p["ARB_IV_MAX"])
    d = arbitrage.repair(grid, iv, weight) - iv
    pred = method(vis, hid) if hid is not None else None
    after = pred + arbitrage.shift_points(grid, d, hid) if hid is not None else None
    return pred, after, iv, d


# --- схемы скрытия и методы -----------------------------------------------------

def mask_configs(obs: pd.DataFrame, p: dict) -> list[tuple]:
    """Схемы пилота (random, wings, slice) плюс far и topn."""
    n = len(obs)
    configs = pilot.mask_configs(obs, p)
    near = np.sort(obs["T"].unique())[: p["FAR_KEEP"]]
    far = np.flatnonzero(~obs["T"].isin(near))
    if 0 < len(far) < n:
        configs.append(("far", np.nan, 0, far))
    order = np.lexsort((-obs["vol_sync"].to_numpy(), -obs["n_candles"].to_numpy()))
    for top in p["TOPN"]:
        if n > top:
            configs.append(("topn", float(top), 0, order[top:]))
    return configs


def make_methods(models: dict, p: dict) -> dict:
    prior = (p["NOISE"], p["TAU_LEVEL"], p["TAU_SLOPE"], p["TAU_SERIES"])
    methods = {
        "hist": lambda vis, hid: hid["hist"].to_numpy(),
        "cb_quad": lambda vis, hid: recon.cb_quad(vis, hid, p["RIDGE"]),
        "cb_svi": lambda vis, hid: recon.cb_svi(vis, hid, p["RIDGE"]),
        "cb_ssvi": lambda vis, hid: ssvi.cb_ssvi(vis, hid, p["SSVI_RIDGE"], p["SSVI_FSCALE"]),
        "hist_cb": lambda vis, hid: recon.hist_cb(vis, hid, prior),
        "moex_today": lambda vis, hid: hid["today"].to_numpy(),
    }
    for name, spec in VARIANTS.items():
        methods[name] = partial(gp.predict, P=models[name], prior=spec["prior"])
    return methods


def _split(obs: pd.DataFrame, hid_idx: np.ndarray):
    mask = np.zeros(len(obs), bool)
    mask[hid_idx] = True
    return obs[~mask], obs[mask].reset_index(drop=True)


def run_day(job: tuple, p: dict, models: dict) -> list[pd.DataFrame]:
    obs, s_prev = job
    obs = obs.reset_index(drop=True)
    methods = make_methods(models, p)
    grid = day_grid(obs, s_prev, p)
    out = []
    for scheme, param, seed, hid_idx in mask_configs(obs, p):
        vis, hid = _split(obs, hid_idx)
        res = hid[KEEP].copy()
        res["scheme"], res["mask"], res["seed"] = scheme, param, seed
        res["n_obs"], res["n_visible"] = len(obs), len(vis)
        res["n_visible_series"] = hid["expiry"].map(vis["expiry"].value_counts()).fillna(0).to_numpy()
        for name, f in methods.items():
            res[name] = f(vis, hid)
        res["gp_af"] = repaired(methods["gp"], vis, hid, grid, p, gp_weight(vis, grid, models["gp"]))[1]
        res["gp_sd"] = gp.predict(vis, hid, models["gp"], return_sd=True)[1]
        out.append(res)
    return out


def run(obs: pd.DataFrame, context: tuple, p: dict, models: dict, workers: int) -> pd.DataFrame:
    """Результаты по скрытым точкам тестового периода для всех схем и методов."""
    smiles, prev, _ = context
    jobs = [(g, smiles[prev[d]]) for d, g in obs[obs["part"] == "test"].groupby("date")]
    with ProcessPoolExecutor(workers) as pool:
        parts = pool.map(partial(run_day, p=p, models=models), jobs, chunksize=4)
        return pd.concat([df for day in parts for df in day], ignore_index=True)


# --- устаревание истории ----------------------------------------------------------

def _lag_day(obs: pd.DataFrame, p: dict, models: dict) -> list[pd.DataFrame]:
    obs = obs.reset_index(drop=True)
    n = len(obs)
    rng = np.random.default_rng(int(obs["date"].iloc[0].strftime("%Y%m%d")) + 1)
    prior = (p["NOISE"], p["TAU_LEVEL"], p["TAU_SLOPE"], p["TAU_SERIES"])
    out = []
    for m in p["LAG_MASKS"]:
        for seed in range(p["LAG_SEEDS"]):
            vis, hid = _split(obs, rng.choice(n, int(np.clip(round(m * n), 1, n - 1)), replace=False))
            for h in p["LAGS"]:
                col = f"hist_lag{h}"
                v, q = vis.assign(hist=vis[col]), hid.assign(hist=hid[col])
                res = hid[["date", "iv"]].copy()
                res["mask"], res["seed"], res["lag"] = m, seed, h
                res["hist"] = q["hist"].to_numpy()
                res["hist_cb"] = recon.hist_cb(v, q, prior)
                res["gp"] = gp.predict(v, q, models[f"gp_lag{h}"])
                res["gp_cb"] = gp.predict(vis, hid, models["gp_cb"], prior=None)
                res["cb_ssvi"] = ssvi.cb_ssvi(vis, hid, p["SSVI_RIDGE"], p["SSVI_FSCALE"])
                out.append(res)
    return out


def run_lags(obs: pd.DataFrame, p: dict, models: dict, workers: int) -> pd.DataFrame:
    """Точность при априоре из кривой h дней назад. Маски одинаковы для всех лагов."""
    cols = [f"hist_lag{h}" for h in p["LAGS"]]
    days = [g for _, g in obs[obs["part"] == "test"].dropna(subset=cols).groupby("date")]
    with ProcessPoolExecutor(workers) as pool:
        parts = pool.map(partial(_lag_day, p=p, models=models), days, chunksize=4)
        res = pd.concat([df for day in parts for df in day], ignore_index=True)
    rows = []
    for (m, h), r in res.groupby(["mask", "lag"]):
        t = pilot.rmse_with_ci(r, ["hist", "hist_cb", "gp", "gp_cb", "cb_ssvi"], p["N_BOOT"])
        rows.append(t.assign(mask=m, lag=h))
    return pd.concat(rows, ignore_index=True)


# --- статический арбитраж ---------------------------------------------------------

def _arb_day(job: tuple, p: dict, models: dict) -> list[dict]:
    obs, s_prev, s_today, F = job
    obs = obs.reset_index(drop=True)
    methods = make_methods(models, p)
    grid = day_grid(obs, s_prev, p, s_today)
    tol = p["ARB_TICKS"] * p["TICK"] / F
    n = len(obs)
    rng = np.random.default_rng(int(obs["date"].iloc[0].strftime("%Y%m%d")) + 2)
    hid_idx = rng.choice(n, int(np.clip(round(p["ARB_MASK"] * n), 1, n - 1)), replace=False)
    rows = []
    for scenario, (vis, hid) in [("full", (obs, None)), ("sparse", _split(obs, hid_idx))]:
        for name in ARB_METHODS:
            weight = gp_weight(vis, grid, models["gp"]) if name == "gp" else None
            pred, after, iv, d = repaired(methods[name], vis, hid, grid, p, weight)
            left = arbitrage.violations(grid, iv + d, tol)
            row = {"date": obs["date"].iloc[0], "scenario": scenario, "method": name, "nodes": len(grid),
                   **arbitrage.violations(grid, iv, tol), "repair_mean": float(np.abs(d).mean()),
                   "repair_max": float(np.abs(d).max()),
                   "left_after": left["butterfly_any"] or left["calendar_any"]}
            if hid is not None:
                zone = arbitrage.shift_points(grid, np.ones(len(grid)), hid) > 0
                row.update(n=len(hid), n_zone=int(zone.sum()), sq_before=float(((pred - hid["iv"]) ** 2).sum()),
                           sq_after=float(((after - hid["iv"]) ** 2).sum()))
            rows.append(row)
    return rows


def run_arbitrage(obs: pd.DataFrame, context: tuple, p: dict, models: dict, workers: int) -> pd.DataFrame:
    """Нарушения арбитража на сетке и размер исправления для каждого метода, по дням теста."""
    smiles, prev, F = context
    jobs = [(g, smiles[prev[d]], smiles[d], float(F[d])) for d, g in obs[obs["part"] == "test"].groupby("date")]
    with ProcessPoolExecutor(workers) as pool:
        parts = pool.map(partial(_arb_day, p=p, models=models), jobs, chunksize=4)
        return pd.DataFrame([r for day in parts for r in day])


def arbitrage_table(arb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, method), g in arb.groupby(["scenario", "method"]):
        row = {"scenario": scenario, "method": method, "days": len(g), "nodes_median": g["nodes"].median(),
               "butterfly_days": g["butterfly_any"].mean(), "calendar_days": g["calendar_any"].mean(),
               "any_days": (g["butterfly_any"] | g["calendar_any"]).mean(),
               "butterfly_share": g["butterfly_share"].mean(), "calendar_share": g["calendar_share"].mean(),
               "repair_mean_vp": 100 * g["repair_mean"].mean(),
               "repair_max_vp": 100 * g.loc[g["repair_max"] > 0, "repair_max"].median(),
               "left_after": g["left_after"].mean()}
        if scenario == "sparse":
            row["rmse_before"] = 100 * np.sqrt(g["sq_before"].sum() / g["n"].sum())
            row["rmse_after"] = 100 * np.sqrt(g["sq_after"].sum() / g["n"].sum())
            row["zone_share"] = g["n_zone"].sum() / g["n"].sum()
        rows.append(row)
    return pd.DataFrame(rows)


# --- пример одного дня (для рисунка) ------------------------------------------------

def example_day(obs: pd.DataFrame, context: tuple, p: dict, models: dict) -> pd.DataFrame:
    """Сделки и восстановленные кривые одного дня теста при доле скрытия EXAMPLE_MASK.

    Строки ``kind``: visible / hidden — сделки; curve — значения методов на сетке по k.
    """
    smiles, prev, _ = context
    date = pd.Timestamp(p["EXAMPLE_DATE"])
    day = obs[obs["date"] == date].reset_index(drop=True)
    n = len(day)
    rng = np.random.default_rng(int(date.strftime("%Y%m%d")) + 3)
    vis, hid = _split(day, rng.choice(n, int(round(p["EXAMPLE_MASK"] * n)), replace=False))
    parts = [vis.assign(kind="visible"), hid.assign(kind="hidden")]
    for e, g in day.groupby("expiry"):
        T = float(g["T"].iloc[0])
        k = np.linspace(g["k"].min(), g["k"].max(), 61)
        c = pd.DataFrame({"date": date, "expiry": e, "T": T, "k": k, "x": k / np.sqrt(T), "sqrtT": np.sqrt(T)})
        c["hist"] = smiles[prev[date]](e, k, T)
        parts.append(c.assign(kind="curve"))
    curves = pd.concat([d for d in parts if d["kind"].iloc[0] == "curve"], ignore_index=True)
    curves["gp"], curves["gp_sd"] = gp.predict(vis, curves, models["gp"], return_sd=True)
    curves["cb_ssvi"] = ssvi.cb_ssvi(vis, curves, p["SSVI_RIDGE"], p["SSVI_FSCALE"])
    cols = ["kind", "date", "expiry", "T", "k", "iv", "n_candles", "hist", "gp", "gp_sd", "cb_ssvi"]
    return pd.concat([parts[0], parts[1], curves], ignore_index=True).reindex(columns=cols)


# --- сводные таблицы --------------------------------------------------------------

def _groups(res: pd.DataFrame):
    for (scheme, m), r in res.groupby(["scheme", res["mask"].fillna(-1)]):
        yield scheme, (np.nan if m < 0 else m), r


def otm_price(k, iv, T):
    """Нормированная цена OTM-опциона (в долях цены фьючерса)."""
    sd = np.asarray(iv, float) * np.sqrt(T)
    d1 = -k / sd + 0.5 * sd
    call = ndtr(d1) - np.exp(k) * ndtr(d1 - sd)
    return np.where(k >= 0, call, call + np.exp(k) - 1.0)


def rmse_table(res: pd.DataFrame, methods: list[str], n_boot: int) -> pd.DataFrame:
    """RMSE с интервалами, MAE и медианная относительная ошибка цены по схемам и методам.

    ``best_cb`` — лучший по RMSE метод только текущей доски в данной схеме.
    """
    rows = []
    for scheme, m, r in _groups(res):
        t = pilot.rmse_with_ci(r, methods, n_boot).set_index("method")
        best = t.loc[CURRENT_BOARD, "rmse"].idxmin()
        visible = float(r.drop_duplicates(["date", "seed"])["n_visible"].median())
        true_price = otm_price(r["k"], r["iv"], r["T"])
        for meth in methods + ["best_cb"]:
            src = best if meth == "best_cb" else meth
            price = otm_price(r["k"], np.maximum(r[src], 1e-3), r["T"])
            rows.append({"scheme": scheme, "mask": m, "method": meth, "source": src,
                         **t.loc[src, ["rmse", "lo", "hi"]].to_dict(),
                         "mae": float(100 * (r[src] - r["iv"]).abs().mean()),
                         "price_err_pct": float(100 * np.median(np.abs(price - true_price) / true_price)),
                         "n_points": len(r), "days": r["date"].nunique(), "visible_median": visible})
    return pd.DataFrame(rows)


def diff_table(res: pd.DataFrame, rmse: pd.DataFrame, pairs: list[tuple[str, str]], n_boot: int) -> pd.DataFrame:
    """Разности RMSE(a) − RMSE(b) с бутстреп-интервалами по дням."""
    b_cb = rmse[rmse["method"] == "best_cb"]
    best = {(s, -1.0 if np.isnan(m) else m): src for s, m, src in zip(b_cb["scheme"], b_cb["mask"], b_cb["source"])}
    rows = []
    for scheme, m, r in _groups(res):
        for a, b in pairs:
            src = best[(scheme, -1.0 if np.isnan(m) else m)] if b == "best_cb" else b
            d, lo, hi = pilot.diff_with_ci(r, a, src, n_boot)
            rows.append({"scheme": scheme, "mask": m, "a": a, "b": b, "source": src, "diff": d, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def bucket_table(res: pd.DataFrame, col: str, bins: list[float], labels: list[str], methods: list[str],
                 mask: float) -> pd.DataFrame:
    """RMSE по корзинам признака ``col`` (схема random, доля скрытия ``mask``)."""
    r = res[(res["scheme"] == "random") & np.isclose(res["mask"], mask)].copy()
    r["bucket"] = pd.cut(r[col], bins, labels=labels)
    rows = []
    for b, g in r.groupby("bucket", observed=True):
        rows.append({"bucket": b, "n": len(g),
                     **{m: float(100 * np.sqrt(((g[m] - g["iv"]) ** 2).mean())) for m in methods}})
    return pd.DataFrame(rows)


def day_move(obs: pd.DataFrame) -> pd.Series:
    """Сдвиг расчётной кривой биржи за день: медиана |today − hist| по сделкам дня, п.п. IV."""
    return (100 * (obs["today"] - obs["hist"]).abs()).groupby(obs["date"]).median()


def regime_table(res: pd.DataFrame, obs: pd.DataFrame, methods: list[str], masks: list[float]) -> pd.DataFrame:
    """RMSE в спокойные, обычные и бурные дни (трети дней теста по сдвигу кривой биржи)."""
    move = day_move(obs[obs["part"] == "test"])
    regime = pd.qcut(move, 3, labels=["calm", "normal", "turbulent"])
    rows = []
    for m in masks:
        r = res[(res["scheme"] == "random") & np.isclose(res["mask"], m)]
        for name, g in r.groupby(r["date"].map(regime), observed=True):
            days = g["date"].unique()
            rows.append({"mask": m, "regime": name, "days": len(days), "move_median": float(move[days].median()),
                         **{meth: float(100 * np.sqrt(((g[meth] - g["iv"]) ** 2).mean())) for meth in methods}})
    return pd.DataFrame(rows)


def series_gap(res: pd.DataFrame, mask: float) -> dict:
    """Где полный метод (gp) проигрывает варианту без вчерашней кривой (gp_cb), схема random.

    Пары «серия, день» сортируются по разности сумм квадратов ошибок. ``worst`` — наименьшее
    число пар, которые дают всю суммарную разность; ``hist_rmse`` — ошибка вчерашней кривой
    в этих парах, п.п.
    """
    r = res[(res["scheme"] == "random") & np.isclose(res["mask"], mask)]
    sq = pd.DataFrame({"date": r["date"], "expiry": r["expiry"], "gp": (r["gp"] - r["iv"]) ** 2,
                       "cb": (r["gp_cb"] - r["iv"]) ** 2, "hist": (r["hist"] - r["iv"]) ** 2, "n": 1})
    g = sq.groupby(["date", "expiry"]).sum()
    g = g.assign(diff=g["gp"] - g["cb"]).sort_values("diff", ascending=False)
    total = g["diff"].sum()
    worst = int((g["diff"].cumsum() < total).sum() + 1) if total > 0 else 0
    w = g.head(worst)
    return {"mask": mask, "total": len(g), "worst": worst,
            "hist_rmse": float(100 * np.sqrt(w["hist"].sum() / w["n"].sum())) if worst else float("nan"),
            "hist_rmse_rest": float(100 * np.sqrt(g["hist"].iloc[worst:].sum() / g["n"].iloc[worst:].sum()))}


def calibration_table(res: pd.DataFrame, P: dict, levels: list[float]) -> pd.DataFrame:
    """Доля скрытых сделок внутри интервала метода: прогноз ± q * sqrt(sd^2 + шум сделки)."""
    rows = []
    for scheme, m, r in _groups(res):
        sd = np.sqrt(r["gp_sd"] ** 2 + gp.noise_var(r["n_candles"].to_numpy(float), P))
        zscore = ((r["iv"] - r["gp"]) / sd).abs()
        rows.append({"scheme": scheme, "mask": m, "sd_median_vp": float(100 * sd.median()),
                     **{f"cover_{int(100 * q)}": float((zscore <= norm.ppf(0.5 + q / 2)).mean()) for q in levels}})
    return pd.DataFrame(rows)


def save(obs: pd.DataFrame, res: pd.DataFrame, models: dict, lags: pd.DataFrame | None,
         arb: pd.DataFrame | None, asset: str, p: dict, data_dir: Path, tables_dir: Path,
         example: pd.DataFrame | None = None) -> dict:
    """Сохраняет результаты по точкам (data/) и сводные таблицы (reports/tables/exp_*)."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(data_dir) / "processed" / f"exp_results_{asset}.csv.gz", index=False)
    methods = BASE_METHODS + [v for v in VARIANTS if v != "gp"]
    rmse = rmse_table(res, methods, p["N_BOOT"])
    rmse.to_csv(tables_dir / f"exp_rmse_{asset}.csv", index=False)
    pairs = [("gp_af", "best_cb"), ("gp_af", "hist"), ("gp_af", "hist_cb"), ("gp_af", "gp"), ("gp_af", "gp_cb"),
             ("gp_af", "moex_today"), ("hist_cb", "best_cb"), ("gp", "hist_cb"),
             *[("gp", v) for v in VARIANTS if v != "gp"]]
    diff_table(res, rmse, pairs, p["N_BOOT"]).to_csv(tables_dir / f"exp_diff_{asset}.csv", index=False)

    m = p["BREAKDOWN_MASK"]
    days = res["T"] * 365
    by = {
        "T": bucket_table(res.assign(days=days), "days", [0, 7, 14, 30, 60, 90, 181],
                          ["3–7", "8–14", "15–30", "31–60", "61–90", "91–180"], BASE_METHODS, m),
        "z": bucket_table(res.assign(az=res["z"].abs()), "az", [-1, 0.5, 1, 2, 100],
                          ["0–0,5", "0,5–1", "1–2", ">2"], BASE_METHODS, m),
        "n": bucket_table(res, "n_candles", [0, 1, 5, 30, 1e6], ["1", "2–5", "6–30", ">30"], BASE_METHODS, m),
    }
    for name, t in by.items():
        t.to_csv(tables_dir / f"exp_by_{name}_{asset}.csv", index=False)
    rr = res[res["scheme"] == "random"].assign(
        bucket=lambda d: pd.cut(d["n_visible_series"], [-0.5, 0.5, 2.5, 4.5, 9.5, 1e9],
                                labels=["0", "1–2", "3–4", "5–9", "10+"]))
    vis = pd.DataFrame([{"bucket": b, "n": len(g), **{k: float(100 * np.sqrt(((g[k] - g["iv"]) ** 2).mean()))
                                                       for k in BASE_METHODS}}
                        for b, g in rr.groupby("bucket", observed=True)])
    vis.to_csv(tables_dir / f"exp_by_visible_{asset}.csv", index=False)
    regime_table(res, obs, BASE_METHODS, p["MASKS"]).to_csv(tables_dir / f"exp_regimes_{asset}.csv", index=False)
    calibration_table(res, models["gp"], p["COVER_LEVELS"]).to_csv(tables_dir / f"exp_calibration_{asset}.csv",
                                                                   index=False)
    if lags is not None:
        lags.to_csv(tables_dir / f"exp_lags_{asset}.csv", index=False)
    if arb is not None:
        arbitrage_table(arb).to_csv(tables_dir / f"exp_arbitrage_{asset}.csv", index=False)
    if example is not None:
        example.to_csv(tables_dir / f"exp_example_{asset}.csv", index=False)
    (tables_dir / f"gp_params_{asset}.json").write_text(json.dumps(models, ensure_ascii=False, indent=2))

    parts = {name: obs[obs["part"] == name] for name in ("train", "test")}
    info = {
        "asset": asset,
        "series_gap": series_gap(res, p["GAP_MASK"]),
        **{f"{name}_period": f"{o['date'].min().date()}..{o['date'].max().date()}" for name, o in parts.items()},
        **{f"{name}_days": int(o["date"].nunique()) for name, o in parts.items()},
        **{f"{name}_points": len(o) for name, o in parts.items()},
        "points_per_day_median": float(parts["test"].groupby("date").size().median()),
        "series_per_day_median": float(parts["test"].groupby("date")["expiry"].nunique().median()),
        "gross_removed": obs.attrs.get("gross_removed"),
        "params": p,
    }
    (tables_dir / f"exp_info_{asset}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info
