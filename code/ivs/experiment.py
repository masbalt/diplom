"""Основной эксперимент: восстановление скрытых сделок при разной разреженности доски.

Отличия от пилота (ivs.pilot):
  * хронологическое разбиение: параметры предлагаемого метода оцениваются на TRAIN_RANGE,
    все числа качества считаются на TEST_RANGE (строго позже);
  * добавлены SSVI и предлагаемый метод (ivs.gp) с упрощёнными вариантами;
  * новые схемы скрытия: ``far`` (видны две ближние серии) и ``topn`` (видны N самых
    торгуемых контрактов дня);
  * предлагаемый метод оценивается и с шагом исправления арбитража (``gp_af``);
  * отдельные исследования: устаревание истории, статический арбитраж, режимы рынка,
    точность оценки неопределённости;
  * динамический вариант метода (``gp_dyn``, ``gp_dyn_af``): начальное приближение — прогноз
    на один день из ivs.forecast вместо вчерашней кривой в координате k;
  * динамический вариант с защитой от грубой ошибки расчётной кривой (``gp_dyn_g``, ``gp_dyn_g_af``);
  * нейросетевой метод сравнения (ivs.neural) с теми же начальными приближениями;
  * контрольный период HOLDOUT_RANGE: дни после тестового периода, по которым не принималось
    ни одного решения. Качество на них считается теми же функциями и пишется в таблицы
    ``exp_*_holdout_*``.

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

from . import arbitrage, forecast, gp, neural, pilot, recon, ssvi

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
    "gp_x": {"prior": "hist_x"},                                              # вчерашняя кривая в координате x
    "gp_dyn": {"prior": "dyn"},                                               # динамическое начальное приближение
}
# Нейросетевой метод сравнения (ivs.neural) с теми же начальными приближениями, что у gp, gp_cb и gp_dyn.
NEURAL = {"np": "hist", "np_cb": None, "np_dyn": "dyn"}
# Начальные приближения из ivs.forecast как самостоятельные методы, динамический вариант с исправлением
# арбитража и он же с защитой от грубой ошибки расчётной кривой (``*_g``, forecast.guard_curves).
EXTRA_METHODS = ["hist_x", "dyn", "gp_dyn_af", *NEURAL, "dyn_g", "gp_dyn_g", "gp_dyn_g_af"]
BREAKDOWN_METHODS = [*BASE_METHODS, "gp_dyn_af"]
ARB_METHODS = ["cb_quad", "cb_svi", "cb_ssvi", "hist", "hist_cb", "gp", "gp_dyn", "moex_today"]
ARB_WEIGHTED = ["gp", "gp_dyn"]   # методы с оценкой неопределённости: исправление арбитража с весами
PARTS = ["test", "holdout"]       # части, на которых считается качество: тестовый и контрольный периоды
KEEP = ["date", "expiry", "strike", "T", "k", "x", "z", "iv", "n_candles"]
SD_FLOOR = 1e-4   # нижняя граница неопределённости метода в узле сетки, доли IV


# --- наблюдения и обучение ------------------------------------------------------

def load(data_dir: Path, asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    board = pd.read_csv(Path(data_dir) / "processed" / f"board_daily_{asset}.csv.gz",
                        parse_dates=["date", "expiry"], low_memory=False)
    sync = pd.read_csv(Path(data_dir) / "processed" / f"trades_sync_{asset}.csv.gz", parse_dates=["date", "expiry"])
    return board, sync


def observations(board: pd.DataFrame, sync: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Наблюдения за TRAIN_RANGE, TEST_RANGE и HOLDOUT_RANGE с колонкой ``part`` (train / test / holdout)."""
    lo, hi = pd.Timestamp(p["TRAIN_RANGE"][0]), pd.Timestamp(p["HOLDOUT_RANGE"][1])
    sync = sync[(sync["date"] >= lo) & (sync["date"] <= hi)]
    obs = pilot.build_observations(board, sync, p, lags=tuple(h for h in p["LAGS"] if h != 1))
    obs["hist_lag1"] = obs["hist"]
    obs["part"] = np.select([obs["date"] >= pd.Timestamp(p["HOLDOUT_RANGE"][0]),
                             obs["date"] >= pd.Timestamp(p["TEST_RANGE"][0])], ["holdout", "test"], "train")
    obs = obs[(obs["part"] != "train") | (obs["date"] <= pd.Timestamp(p["TRAIN_RANGE"][1]))]
    return obs.reset_index(drop=True)


def _fit_variant(job: tuple) -> tuple[str, dict]:
    name, days, spec = job
    P = gp.fit(days, spec.get("init", gp.INIT), spec.get("fixed"), spec.get("flags"))
    return name, P


def train(obs: pd.DataFrame, p: dict, workers: int) -> dict:
    """Параметры всех вариантов метода по дням обучающего периода (в том числе для лагов истории)
    и веса нейросетевого метода сравнения (ключ ``np``)."""
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
        models = dict(pool.map(_fit_variant, jobs))
    models["np"] = neural.fit_all(tr, NEURAL, p, workers)
    return models


# --- вчерашние кривые и сетка дня -----------------------------------------------

def day_context(board: pd.DataFrame) -> tuple[dict, dict, pd.Series]:
    """Расчётные кривые по дням, предыдущий торговый день и медианная цена фьючерса дня."""
    dates = sorted(board["date"].unique())
    return pilot.settle_smiles(board), dict(zip(dates[1:], dates[:-1])), board.groupby("date")["F"].median()


def day_grid(obs: pd.DataFrame, s_prev, p: dict, s_today=None, dyn: tuple | None = None) -> pd.DataFrame:
    """Сетка для проверки арбитража по сериям дня; в узлах — вчерашняя (и сегодняшняя) кривая.

    ``dyn`` — аргументы forecast.dyn_prior (сделки предыдущего дня, ядро разности, доля переноса):
    тогда в узлах считаются ещё начальные приближения ``hist_x`` и ``dyn``.
    """
    ser = obs[["expiry", "T"]].drop_duplicates().copy()
    ser["atm"] = [float(s_prev(e, np.array([0.0]), T)[0]) for e, T in zip(ser["expiry"], ser["T"])]
    grid = arbitrage.make_grid(ser, p["ARB_Z_MAX"], p["ARB_STEP"])
    for col, smiles in (("hist", s_prev), ("today", s_today)):
        if smiles is not None:
            grid[col] = np.nan
            for e, g in grid.groupby("expiry"):
                grid.loc[g.index, col] = smiles(e, g["k"].to_numpy(), float(g["T"].iloc[0]))
    if dyn is not None:
        grid["hist_x"], grid["dyn"] = forecast.dyn_prior(grid, s_prev, *dyn)
    return grid


def dyn_args(models: dict, vis_prev: pd.DataFrame | None) -> tuple:
    """Аргументы forecast.dyn_prior для дня: сделки предыдущего дня, ядро разности, доля переноса на один день."""
    return vis_prev, models["fc"]["resid"], models["fc"]["phi"][1]


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
        "hist_x": lambda vis, hid: hid["hist_x"].to_numpy(),
        "dyn": lambda vis, hid: hid["dyn"].to_numpy(),
        "dyn_g": lambda vis, hid: hid["dyn_g"].to_numpy(),
    }
    for name, spec in VARIANTS.items():
        methods[name] = partial(gp.predict, P=models[name], prior=spec["prior"])
    # Динамический вариант с защитой: то же ядро, начальное приближение по кривым после forecast.guard_curves.
    methods["gp_dyn_g"] = partial(gp.predict, P=models["gp_dyn"], prior="dyn_g")
    for name, model in models["np"].items():
        methods[name] = partial(neural.predict, model=model, name=name)
    return methods


def _split(obs: pd.DataFrame, hid_idx: np.ndarray):
    mask = np.zeros(len(obs), bool)
    mask[hid_idx] = True
    return obs[~mask], obs[mask].reset_index(drop=True)


def run_day(job: tuple, p: dict, models: dict) -> list[pd.DataFrame]:
    obs, s_prev, vis_prev, s_guard = job
    obs = obs.reset_index(drop=True)
    methods = make_methods(models, p)
    grid = day_grid(obs, s_prev, p, dyn=dyn_args(models, vis_prev))
    # Сетка для варианта с защитой нужна только в дни, когда защита убрала серию из вчерашних кривых.
    grid_g = None if s_guard is None else day_grid(obs, s_guard, p, dyn=dyn_args(models, vis_prev)).rename(
        columns={"dyn": "dyn_g"})
    out = []
    for scheme, param, seed, hid_idx in mask_configs(obs, p):
        vis, hid = _split(obs, hid_idx)
        res = hid[KEEP].copy()
        res["part"] = obs["part"].iloc[0]
        res["scheme"], res["mask"], res["seed"] = scheme, param, seed
        res["n_obs"], res["n_visible"] = len(obs), len(vis)
        res["n_visible_series"] = hid["expiry"].map(vis["expiry"].value_counts()).fillna(0).to_numpy()
        for name, f in methods.items():
            res[name] = f(vis, hid)
        for name in ARB_WEIGHTED:
            res[f"{name}_af"] = repaired(methods[name], vis, hid, grid, p, gp_weight(vis, grid, models[name]))[1]
        res["gp_dyn_g_af"] = res["gp_dyn_af"] if grid_g is None else repaired(
            methods["gp_dyn_g"], vis, hid, grid_g, p, gp_weight(vis, grid_g, models["gp_dyn"]))[1]
        res["gp_sd"] = gp.predict(vis, hid, models["gp"], return_sd=True)[1]
        out.append(res)
    return out


def run(obs: pd.DataFrame, context: tuple, p: dict, models: dict, workers: int,
        guarded: dict | None = None) -> pd.DataFrame:
    """Результаты по скрытым точкам тестового и контрольного периодов для всех схем и методов.

    ``guarded`` — кривые по дням после forecast.guard_curves (для варианта с защитой)."""
    smiles, prev, _ = context
    by_day = dict(tuple(obs.groupby("date")))
    guarded = guarded or smiles
    jobs = [(g, smiles[prev[d]], by_day.get(prev[d]), None if guarded[prev[d]] is smiles[prev[d]] else guarded[prev[d]])
            for d, g in by_day.items() if g["part"].iloc[0] in PARTS]
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
    obs, s_prev, s_today, F, vis_prev = job
    obs = obs.reset_index(drop=True)
    methods = make_methods(models, p)
    grid = day_grid(obs, s_prev, p, s_today, dyn_args(models, vis_prev))
    tol = p["ARB_TICKS"] * p["TICK"] / F
    n = len(obs)
    rng = np.random.default_rng(int(obs["date"].iloc[0].strftime("%Y%m%d")) + 2)
    hid_idx = rng.choice(n, int(np.clip(round(p["ARB_MASK"] * n), 1, n - 1)), replace=False)
    rows = []
    for scenario, (vis, hid) in [("full", (obs, None)), ("sparse", _split(obs, hid_idx))]:
        for name in ARB_METHODS:
            weight = gp_weight(vis, grid, models[name]) if name in ARB_WEIGHTED else None
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
    by_day = dict(tuple(obs.groupby("date")))
    jobs = [(g, smiles[prev[d]], smiles[d], float(F[d]), by_day.get(prev[d]))
            for d, g in by_day.items() if g["part"].iloc[0] == "test"]
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


def series_gap(res: pd.DataFrame, mask: float, method: str = "gp") -> dict:
    """Где метод ``method`` проигрывает варианту без вчерашней кривой (gp_cb), схема random.

    Пары «серия, день» сортируются по разности сумм квадратов ошибок. ``worst`` — наименьшее
    число пар, которые дают всю суммарную разность (0, если метод не проигрывает);
    ``hist_rmse`` — ошибка вчерашней кривой в этих парах, п.п.
    """
    r = res[(res["scheme"] == "random") & np.isclose(res["mask"], mask)]
    sq = pd.DataFrame({"date": r["date"], "expiry": r["expiry"], "gp": (r[method] - r["iv"]) ** 2,
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


def worst_prior_pair(res: pd.DataFrame) -> dict:
    """Пара «серия, день», которая даёт наибольшую сумму квадратов ошибки вчерашней кривой.

    Считается по разным сделкам (каждая сделка один раз, независимо от схем скрытия).
    ``share_sse`` — доля этой пары в сумме квадратов ошибки вчерашней кривой по всем сделкам.
    """
    u = res.drop_duplicates(["date", "expiry", "strike"])
    sq = ((u["hist"] - u["iv"]) ** 2).groupby([u["date"], u["expiry"]]).agg(["sum", "size"])
    date, expiry = sq["sum"].idxmax()
    worst = sq.loc[(date, expiry)]
    return {"date": str(pd.Timestamp(date).date()), "expiry": str(pd.Timestamp(expiry).date()),
            "trades": int(worst["size"]), "hist_err_vp": float(100 * np.sqrt(worst["sum"] / worst["size"])),
            "share_sse": float(worst["sum"] / sq["sum"].sum()), "pairs": len(sq)}


def summary_tables(res: pd.DataFrame, asset: str, p: dict, tables_dir: Path) -> None:
    """RMSE и разности методов по сценариям для тестового и контрольного периодов.

    Таблицы зависят только от результатов по скрытым сделкам, поэтому пересчитываются без
    повторного расчёта методов (этап ``summary`` в main.py). Для контрольного периода дополнительно
    пишутся таблицы без одной пары «серия, день» с наибольшей ошибкой вчерашней кривой
    (``exp_*_holdout_trim_*``) и описание этой пары (``exp_holdout_outlier_*.json``): так видно,
    насколько вывод зависит от единичной ошибки расчётной кривой биржи.
    """
    methods = BASE_METHODS + [v for v in VARIANTS if v != "gp"] + EXTRA_METHODS
    pairs = [("gp_af", "best_cb"), ("gp_af", "hist"), ("gp_af", "hist_cb"), ("gp_af", "gp"), ("gp_af", "gp_cb"),
             ("gp_af", "moex_today"), ("hist_cb", "best_cb"), ("gp", "hist_cb"),
             *[("gp", v) for v in VARIANTS if v != "gp"],
             ("gp_dyn_af", "gp_af"), ("gp_dyn_af", "best_cb"), ("gp_dyn_af", "hist"), ("gp_dyn_af", "dyn"),
             ("gp_dyn_af", "gp_dyn"), ("gp_dyn_af", "gp_cb"), ("gp_dyn_af", "moex_today"), ("gp_dyn", "gp_cb"),
             ("gp_dyn", "hist"), ("gp_dyn", "dyn"), ("np", "gp"), ("np_cb", "gp_cb"), ("np_dyn", "gp_dyn"),
             ("gp_dyn_af", "np_dyn"), ("np_dyn", "best_cb"), ("np_dyn", "hist"), ("np_dyn", "dyn"),
             ("gp_dyn_g_af", "gp_dyn_af"), ("gp_dyn_g_af", "gp_af"), ("gp_dyn_g_af", "best_cb"),
             ("gp_dyn_g_af", "hist"), ("gp_dyn_g_af", "moex_today")]

    def write(r: pd.DataFrame, tag: str) -> None:
        rmse = rmse_table(r, methods, p["N_BOOT"])
        rmse.to_csv(tables_dir / f"exp_rmse_{tag}.csv", index=False)
        diff_table(r, rmse, pairs, p["N_BOOT"]).to_csv(tables_dir / f"exp_diff_{tag}.csv", index=False)

    for part in PARTS:
        r = res[res["part"] == part]
        if len(r):
            write(r, asset if part == "test" else f"{part}_{asset}")
    hold = res[res["part"] == "holdout"]
    if len(hold):
        worst = worst_prior_pair(hold)
        same = (hold["date"] == pd.Timestamp(worst["date"])) & (hold["expiry"] == pd.Timestamp(worst["expiry"]))
        write(hold[~same], f"holdout_trim_{asset}")
        (tables_dir / f"exp_holdout_outlier_{asset}.json").write_text(json.dumps(worst, ensure_ascii=False, indent=2))


def save(obs: pd.DataFrame, res: pd.DataFrame, models: dict, lags: pd.DataFrame | None,
         arb: pd.DataFrame | None, asset: str, p: dict, data_dir: Path, tables_dir: Path,
         example: pd.DataFrame | None = None) -> dict:
    """Сохраняет результаты по точкам (data/) и сводные таблицы (reports/tables/exp_*).

    Все таблицы считаются на тестовом периоде. Для контрольного периода пишутся только RMSE и
    разности методов: ``exp_rmse_holdout_{asset}.csv`` и ``exp_diff_holdout_{asset}.csv``.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(data_dir) / "processed" / f"exp_results_{asset}.csv.gz", index=False)
    summary_tables(res, asset, p, tables_dir)
    res = res[res["part"] == "test"]

    m = p["BREAKDOWN_MASK"]
    days = res["T"] * 365
    by = {
        "T": bucket_table(res.assign(days=days), "days", [0, 7, 14, 30, 60, 90, 181],
                          ["3–7", "8–14", "15–30", "31–60", "61–90", "91–180"], BREAKDOWN_METHODS, m),
        "z": bucket_table(res.assign(az=res["z"].abs()), "az", [-1, 0.5, 1, 2, 100],
                          ["0–0,5", "0,5–1", "1–2", ">2"], BREAKDOWN_METHODS, m),
        "n": bucket_table(res, "n_candles", [0, 1, 5, 30, 1e6], ["1", "2–5", "6–30", ">30"], BREAKDOWN_METHODS, m),
    }
    for name, t in by.items():
        t.to_csv(tables_dir / f"exp_by_{name}_{asset}.csv", index=False)
    rr = res[res["scheme"] == "random"].assign(
        bucket=lambda d: pd.cut(d["n_visible_series"], [-0.5, 0.5, 2.5, 4.5, 9.5, 1e9],
                                labels=["0", "1–2", "3–4", "5–9", "10+"]))
    vis = pd.DataFrame([{"bucket": b, "n": len(g), **{k: float(100 * np.sqrt(((g[k] - g["iv"]) ** 2).mean()))
                                                       for k in BREAKDOWN_METHODS}}
                        for b, g in rr.groupby("bucket", observed=True)])
    vis.to_csv(tables_dir / f"exp_by_visible_{asset}.csv", index=False)
    regime_table(res, obs, BREAKDOWN_METHODS, p["MASKS"]).to_csv(tables_dir / f"exp_regimes_{asset}.csv", index=False)
    calibration_table(res, models["gp"], p["COVER_LEVELS"]).to_csv(tables_dir / f"exp_calibration_{asset}.csv",
                                                                   index=False)
    if lags is not None:
        lags.to_csv(tables_dir / f"exp_lags_{asset}.csv", index=False)
    if arb is not None:
        arbitrage_table(arb).to_csv(tables_dir / f"exp_arbitrage_{asset}.csv", index=False)
    if example is not None:
        example.to_csv(tables_dir / f"exp_example_{asset}.csv", index=False)
    saved = {k: v for k, v in models.items() if k != "np"}   # веса сетей восстанавливаются обучением с теми же seed
    (tables_dir / f"gp_params_{asset}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2))

    parts = {name: obs[obs["part"] == name] for name in ("train", *PARTS) if (obs["part"] == name).any()}
    info = {
        "asset": asset,
        "series_gap": series_gap(res, p["GAP_MASK"]),
        "series_gap_dyn": series_gap(res, p["GAP_MASK"], "gp_dyn"),
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
