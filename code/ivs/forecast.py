"""Прогноз поверхности на h торговых дней и прогноз как начальное приближение для восстановления.

Прогноз строится в конце дня t для дня d = t + h и использует только данные не позже дня t.
Качество считается на всех сделках дня d: из дня d в прогноз не попадает ничего.

Правила переноса расчётной кривой биржи дня t (обучения не требуют):
  fc_k      — волатильность читается при той же денежности k = ln(K/F)
              (так устроена «вчерашняя кривая» основного эксперимента);
  fc_strike — при том же страйке K;
  fc_x      — при той же нормированной денежности x = k / sqrt(T).
Модели добавляют к fc_x прогноз изменения, их параметры оцениваются на обучающем периоде:
  fc_ar     — авторегрессия пяти факторов общей поверхности дня (возврат к скользящему среднему);
  fc_gbm    — градиентный бустинг изменения уровня серии по признакам дня t;
  fc_carry  — перенос поправки: доля phi_h сглаженной разности между сделками дня t
              и расчётной кривой того же дня.

Динамическое начальное приближение (колонка ``dyn``) — это fc_carry при h = 1. Предлагаемый метод
(ivs.gp) с ним вместо вчерашней кривой в координате k — вариант ``gp_dyn`` основного эксперимента.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import HistGradientBoostingRegressor

from . import gp, pilot

RULES = {"fc_k": "k", "fc_strike": "strike", "fc_x": "x"}
MODELS = ["fc_ar", "fc_gbm", "fc_carry"]
METHODS = [*RULES, *MODELS]
PAIRS = [("fc_strike", "fc_k"), ("fc_x", "fc_k"), ("fc_ar", "fc_k"), ("fc_gbm", "fc_k"), ("fc_carry", "fc_k"),
         ("fc_ar", "fc_x"), ("fc_gbm", "fc_x"), ("fc_carry", "fc_x")]
KEEP = ["date", "expiry", "strike", "T", "k", "x", "sqrtT", "z", "iv", "today", "n_candles", "part"]
X_GRID = np.linspace(-0.6, 0.6, 13)   # узлы по x для факторов общей поверхности дня
SKEW_X = 0.3                          # наклон и кривизна серии считаются по точкам x = -0.3, 0, 0.3
FEATURES = ["chg1", "chg5", "dev10", "slope_near", "slope_far", "rv_gap", "ret", "absret", "med_dev20",
            "med_chg1", "med_chg5", "sqrtT", "invT", "lvl", "skew", "curv"]


def calendar(board: pd.DataFrame) -> dict:
    """Торговые дни, их номера и цена фьючерса по сериям (нужна для переноса по страйку)."""
    dates = sorted(board["date"].unique())
    F = board.groupby(["date", "expiry"])["F"].median()
    return {"dates": dates, "pos": {d: i for i, d in enumerate(dates)},
            "F": {d: s.droplevel(0).to_dict() for d, s in F.groupby(level=0)}}


# --- правила переноса и перенос поправки ---------------------------------------------

def transfer(sm, expiry, k: np.ndarray, T: float, rule: str = "x", strike=None, F=None) -> np.ndarray:
    """Кривая серии ``expiry`` из набора кривых ``sm`` дня t в точках дня d: денежность k, срок T.

    Серии, которой в день t ещё не было, соответствует ближайшая по сроку серия в координате x.
    """
    if rule == "k" or expiry not in sm.series or (rule == "strike" and F is None):
        return sm(expiry, k, T)
    Ts, ks, iv = sm.series[expiry]
    return np.interp(k * np.sqrt(Ts / T) if rule == "x" else np.log(strike / F), ks, iv)


def smooth_resid(vis: pd.DataFrame, pts: pd.DataFrame, P: dict) -> np.ndarray:
    """Сглаженная разность «сделка − расчётная кривая того же дня» (сделки ``vis``) в точках ``pts``.

    Сглаживание — апостериорное среднее гауссовского процесса с ядром ivs.gp и параметрами ``P``.
    """
    A, B = gp.arrays(vis, "today"), gp.arrays(pts, None)
    c = cho_factor(gp.kernel(A, A, P) + np.diag(gp.noise_var(A["n"], P)), lower=True)
    return gp.kernel(B, A, P) @ cho_solve(c, A["iv"] - A["prior"])


def fit_resid(obs: pd.DataFrame) -> dict:
    """Параметры ядра для разности «сделка − кривая того же дня» по дням обучающего периода."""
    return gp.fit([gp.arrays(g, "today") for _, g in obs[obs["part"] == "train"].groupby("date")])


def by_series(pts: pd.DataFrame, f) -> np.ndarray:
    """Значения ``f(expiry, строки серии, срок серии)`` в порядке строк ``pts``."""
    out = np.full(len(pts), np.nan)
    for e, g in pts.groupby("expiry", sort=False):
        out[pts.index.get_indexer(g.index)] = f(e, g, float(g["T"].iloc[0]))
    return out


def dyn_prior(pts: pd.DataFrame, sm, vis: pd.DataFrame | None, P: dict, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Вчерашняя кривая в координате x и динамическое начальное приближение в точках ``pts``.

    ``sm`` — кривые предыдущего дня, ``vis`` — сделки предыдущего дня (может не быть).
    """
    hist_x = by_series(pts, lambda e, g, T: transfer(sm, e, g["k"].to_numpy(), T, "x"))
    if vis is None or not len(vis):
        return hist_x, hist_x.copy()
    return hist_x, hist_x + phi * smooth_resid(vis, pts, P)


def add_prior(obs: pd.DataFrame, smiles: dict, prev: dict, P: dict, phi: float) -> pd.DataFrame:
    """Колонки ``hist_x`` и ``dyn`` для каждой сделки: начальные приближения из данных предыдущего дня."""
    obs = obs.reset_index(drop=True)
    by_day = dict(tuple(obs.groupby("date")))
    hist_x, dyn = np.full(len(obs), np.nan), np.full(len(obs), np.nan)
    for d, q in by_day.items():
        hist_x[q.index], dyn[q.index] = dyn_prior(q, smiles[prev[d]], by_day.get(prev[d]), P, phi)
    return obs.assign(hist_x=hist_x, dyn=dyn)


# --- прогнозы правил для всех дней и горизонтов --------------------------------------

def _target_day(job: tuple, P: dict) -> pd.DataFrame:
    """Прогнозы для сделок одного дня d при всех горизонтах: правила переноса и сглаженная разность."""
    q, origins = job
    q = q.reset_index(drop=True)
    parts = []
    for h, (t, sm, vis, F) in origins.items():
        out = q[KEEP].copy()
        out["h"], out["origin"] = h, t
        for name, rule in RULES.items():
            out[name] = by_series(q, lambda e, g, T: transfer(sm, e, g["k"].to_numpy(), T, rule,
                                                              g["strike"].to_numpy(), F.get(e)))
        out["corr"] = smooth_resid(vis, q, P) if vis is not None and len(vis) else 0.0
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def frames(obs: pd.DataFrame, smiles: dict, cal: dict, lags: list[int], P: dict, workers: int) -> pd.DataFrame:
    """Строка на сделку дня d и горизонт h: прогнозы правил и сглаженная разность дня t = d − h."""
    by_day = dict(tuple(obs.groupby("date")))
    jobs = []
    for d, q in by_day.items():
        origins = {}
        for h in lags:
            i = cal["pos"][d] - h
            if i >= 0 and cal["dates"][i] in smiles:
                t = cal["dates"][i]
                origins[h] = (t, smiles[t], by_day.get(t), cal["F"].get(t, {}))
        if origins:
            jobs.append((q, origins))
    with ProcessPoolExecutor(workers) as pool:
        return pd.concat(pool.map(partial(_target_day, P=P), jobs, chunksize=4), ignore_index=True)


# --- модели изменения поверхности ------------------------------------------------------

def _basis(x, sqrtT) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x, x**2, sqrtT, x * sqrtT])


def factor_table(smiles: dict, p: dict) -> pd.DataFrame:
    """Факторы дня: коэффициенты поверхности b0 + b1 x + b2 x^2 + b3 sqrt(T) + b4 x sqrt(T),
    подогнанной к расчётным кривым биржи всех серий дня."""
    rows = {}
    for d, sm in smiles.items():
        X, y = [], []
        for T, ks, iv in sm.series.values():
            k = X_GRID * np.sqrt(T)
            ok = (k >= ks.min()) & (k <= ks.max())
            if p["T_MIN_DAYS"] <= 365 * T <= p["T_MAX_DAYS"] and ok.sum() >= 5:
                X.append(_basis(X_GRID[ok], np.full(ok.sum(), np.sqrt(T))))
                y.append(np.interp(k[ok], ks, iv))
        rows[d] = np.linalg.lstsq(np.vstack(X), np.concatenate(y), rcond=None)[0] if X else np.full(5, np.nan)
    return pd.DataFrame(rows).T.sort_index()


def ar_beta(dev: pd.DataFrame, fac: pd.DataFrame, h: int, train_end: pd.Timestamp) -> np.ndarray:
    """Коэффициенты прямой регрессии: изменение фактора за h дней = beta * отклонение от среднего."""
    chg = fac.shift(-h) - fac
    ok = np.asarray(fac.index.to_series().shift(-h) <= train_end)   # день прогноза внутри обучения
    num = (dev[ok] * chg[ok]).sum(skipna=True)
    return (num / (dev[ok] ** 2).where(chg[ok].notna()).sum(skipna=True)).to_numpy()


def level_table(smiles: dict, cal: dict, p: dict) -> pd.DataFrame:
    """Строка на серию и день: уровень (IV на деньгах), наклон и кривизна расчётной кривой и
    признаки для прогноза изменения уровня. Всё считается по данным не позже этого дня."""
    rows = []
    for d in cal["dates"]:
        for e, (T, ks, iv) in (smiles[d].series.items() if d in smiles else ()):
            if 365 * T >= 2 and ks.min() < 0 < ks.max():
                lo, mid, hi = np.interp(np.array([-SKEW_X, 0.0, SKEW_X]) * np.sqrt(T), ks, iv)
                rows.append({"date": d, "expiry": e, "T": T, "lvl": mid, "skew": (hi - lo) / (2 * SKEW_X),
                             "curv": (hi + lo - 2 * mid) / SKEW_X**2})
    L = pd.DataFrame(rows)
    F = pd.DataFrame([(d, e, f) for d, s in cal["F"].items() for e, f in s.items()],
                     columns=["date", "expiry", "F"]).sort_values(["expiry", "date"])
    F["ret"] = np.log(F["F"]).groupby(F["expiry"]).diff()
    day = pd.DataFrame({"ret": F.groupby("date")["ret"].median()})   # доходность фьючерса за день
    day["absret"] = day["ret"].abs()
    day["rv10"] = day["ret"].rolling(10).std() * np.sqrt(252)
    by_T = L.sort_values("T").groupby("date")["lvl"]
    day["near"], day["far"], day["med"] = by_T.first(), by_T.last(), by_T.median()
    day["med_dev20"] = day["med"] - day["med"].rolling(20, min_periods=5).mean()
    day["med_chg1"], day["med_chg5"] = day["med"].diff(), day["med"].diff(5)
    L = L.merge(day.reset_index(), on="date", how="left").sort_values(["expiry", "date"]).reset_index(drop=True)
    lvl = L.groupby("expiry")["lvl"]
    L["chg1"], L["chg5"] = lvl.diff(), lvl.diff(5)
    L["dev10"] = L["lvl"] - lvl.transform(lambda s: s.rolling(10, min_periods=3).mean())
    L["slope_near"], L["slope_far"] = L["lvl"] - L["near"], L["lvl"] - L["far"]
    L["rv_gap"] = L["rv10"] - L["lvl"]
    L["sqrtT"], L["invT"] = np.sqrt(L["T"]), 1.0 / np.sqrt(L["T"])
    return L


def gbm_shift(L: pd.DataFrame, cal: dict, h: int, train_end: pd.Timestamp, params: dict) -> pd.Series:
    """Прогноз изменения уровня серии за h дней для каждой пары (день, серия)."""
    pos, dates = cal["pos"], cal["dates"]
    target = L["date"].map(lambda d: dates[pos[d] + h] if pos[d] + h < len(dates) else pd.NaT)
    future = L.set_index(["date", "expiry"])["lvl"]
    y = future.reindex(pd.MultiIndex.from_arrays([target, L["expiry"]])).to_numpy() - L["lvl"].to_numpy()
    fit = np.isfinite(y) & np.asarray(target <= train_end)
    model = HistGradientBoostingRegressor(random_state=0, **params).fit(L.loc[fit, FEATURES], y[fit])
    return pd.Series(model.predict(L[FEATURES]), index=pd.MultiIndex.from_frame(L[["date", "expiry"]]))


def fit_apply(fr: pd.DataFrame, smiles: dict, cal: dict, p: dict) -> tuple[pd.DataFrame, dict]:
    """Оценивает модели на обучающих днях и добавляет прогнозы fc_ar, fc_gbm, fc_carry.

    В параметрах: ``phi`` — доля переноса поправки для каждого горизонта; ``phi_curve`` — её часть,
    которая объясняется сдвигом самой расчётной кривой биржи за h дней; ``ar_beta`` — коэффициенты
    авторегрессии факторов.
    """
    train_end = pd.Timestamp(p["TRAIN_RANGE"][1])
    fac = factor_table(smiles, p)
    dev = fac - fac.rolling(p["FC_AR_WINDOW"], min_periods=p["FC_AR_WINDOW"] // 3).mean()
    L = level_table(smiles, cal, p)
    models = {"phi": {}, "phi_curve": {}, "ar_beta": {}}
    parts = []
    for h, r in fr.groupby("h"):
        r = r.copy()
        tr = r[r["part"] == "train"]
        phi = float((tr["corr"] * (tr["iv"] - tr["fc_x"])).sum() / (tr["corr"] ** 2).sum())
        # Какая часть phi приходится на сдвиг самой расчётной кривой за h дней (кривая дня t + h минус кривая дня t).
        models["phi_curve"][int(h)] = float((tr["corr"] * (tr["today"] - tr["fc_x"])).sum() / (tr["corr"] ** 2).sum())
        beta = ar_beta(dev, fac, h, train_end)
        d_fac = np.nan_to_num(dev.reindex(r["origin"]).to_numpy() * beta)
        r["fc_ar"] = r["fc_x"] + (_basis(r["x"].to_numpy(), r["sqrtT"].to_numpy()) * d_fac).sum(axis=1)
        shift = gbm_shift(L, cal, h, train_end, p["FC_GBM"])
        r["fc_gbm"] = r["fc_x"] + shift.reindex(pd.MultiIndex.from_frame(r[["origin", "expiry"]])).fillna(0.0).to_numpy()
        r["fc_carry"] = r["fc_x"] + phi * r["corr"]
        models["phi"][int(h)], models["ar_beta"][int(h)] = phi, [float(b) for b in beta]
        parts.append(r)
    return pd.concat(parts, ignore_index=True), models


# --- сводные таблицы ---------------------------------------------------------------------

def rmse_table(fr: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """RMSE прогнозов на сделках дня d с интервалами бутстрепом по дням, по горизонтам."""
    return pd.concat([pilot.rmse_with_ci(r, METHODS, n_boot).assign(h=h, n_points=len(r), days=r["date"].nunique())
                      for h, r in fr.groupby("h")], ignore_index=True)


def diff_table(fr: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for h, r in fr.groupby("h"):
        for a, b in PAIRS:
            d, lo, hi = pilot.diff_with_ci(r, a, b, n_boot)
            rows.append({"h": h, "a": a, "b": b, "diff": d, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def bucket_table(fr: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """RMSE правил переноса по сроку серии и по денежности |z| для выбранных горизонтов."""
    specs = {"T": (fr["T"] * 365, [0, 7, 14, 30, 60, 181], ["3–7", "8–14", "15–30", "31–60", "61–180"]),
             "z": (fr["z"].abs(), [-1, 0.5, 1, 2, 100], ["0–0,5", "0,5–1", "1–2", ">2"])}
    rows = []
    for by, (values, bins, labels) in specs.items():
        bucket = pd.cut(values, bins, labels=labels)
        for (h, b), g in fr[fr["h"].isin(horizons)].groupby(["h", bucket], observed=True):
            rows.append({"by": by, "h": h, "bucket": b, "n": len(g),
                         **{m: float(100 * np.sqrt(((g[m] - g["iv"]) ** 2).mean())) for m in METHODS}})
    return pd.DataFrame(rows)


# --- полный расчёт и сохранение -----------------------------------------------------------

def run(obs: pd.DataFrame, smiles: dict, prev: dict, cal: dict, p: dict, workers: int):
    """Оценивает модели на обучающем периоде и считает прогнозы на остальных днях.

    Возвращает (наблюдения с колонками ``hist_x`` и ``dyn``, прогнозы по сделкам тестового и
    контрольного периодов, параметры моделей). ``resid`` в параметрах — ядро для разности
    «сделка − кривая того же дня», ``phi`` — доля переноса поправки для каждого горизонта.
    """
    P = fit_resid(obs)
    fr, models = fit_apply(frames(obs, smiles, cal, p["LAGS"], P, workers), smiles, cal, p)
    models["resid"] = P
    out = add_prior(obs, smiles, prev, P, models["phi"][1])
    out.attrs = dict(obs.attrs)
    return out, fr[fr["part"] != "train"].reset_index(drop=True), models


def save(fr: pd.DataFrame, models: dict, asset: str, p: dict, data_dir, tables_dir) -> None:
    """Прогнозы по сделкам (data/), сводные таблицы и параметры моделей ``reports/tables/exp_forecast_*``."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    fr.to_csv(data_dir / "processed" / f"exp_forecast_results_{asset}.csv.gz", index=False)
    (tables_dir / f"exp_forecast_params_{asset}.json").write_text(json.dumps(models, ensure_ascii=False, indent=2))
    for part in ("test", "holdout"):
        r = fr[fr["part"] == part]
        if len(r):
            tag = asset if part == "test" else f"{part}_{asset}"
            rmse_table(r, p["N_BOOT"]).to_csv(tables_dir / f"exp_forecast_{tag}.csv", index=False)
            diff_table(r, p["N_BOOT"]).to_csv(tables_dir / f"exp_forecast_diff_{tag}.csv", index=False)
    bucket_table(fr[fr["part"] == "test"], p["FC_BUCKET_LAGS"]).to_csv(
        tables_dir / f"exp_forecast_buckets_{asset}.csv", index=False)
