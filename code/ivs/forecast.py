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
  fc_nn     — небольшая нейросеть с теми же признаками и той же целью, что у бустинга;
  fc_carry  — перенос поправки: доля phi_h сглаженной разности между сделками дня t
              и расчётной кривой того же дня.

Динамическое начальное приближение (колонка ``dyn``) — это fc_carry при h = 1. Предлагаемый метод
(ivs.gp) с ним вместо вчерашней кривой в координате k — вариант ``gp_dyn`` основного эксперимента.

Защита от грубой ошибки расчётной кривой (``guard_curves``): кривая серии без сделок, которая
отстоит от соседних серий со сделками больше чем на долю GUARD_GAP их уровня, в историю не
берётся. Начальное приближение с защитой — колонка ``dyn_g``.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import copy

import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import HistGradientBoostingRegressor

from . import gp, pilot

RULES = {"fc_k": "k", "fc_strike": "strike", "fc_x": "x"}
MODELS = ["fc_ar", "fc_gbm", "fc_nn", "fc_carry"]
METHODS = [*RULES, *MODELS]
PAIRS = [("fc_strike", "fc_k"), ("fc_x", "fc_k"), ("fc_ar", "fc_k"), ("fc_gbm", "fc_k"), ("fc_nn", "fc_k"),
         ("fc_carry", "fc_k"), ("fc_ar", "fc_x"), ("fc_gbm", "fc_x"), ("fc_nn", "fc_x"), ("fc_carry", "fc_x")]
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


def add_prior(obs: pd.DataFrame, smiles: dict, prev: dict, P: dict, phi: float,
              names: tuple[str, str] = ("hist_x", "dyn")) -> pd.DataFrame:
    """Колонки ``names`` для каждой сделки: вчерашняя кривая в координате x и динамическое начальное
    приближение, оба из данных предыдущего дня."""
    obs = obs.reset_index(drop=True)
    by_day = dict(tuple(obs.groupby("date")))
    hist_x, dyn = np.full(len(obs), np.nan), np.full(len(obs), np.nan)
    for d, q in by_day.items():
        hist_x[q.index], dyn[q.index] = dyn_prior(q, smiles[prev[d]], by_day.get(prev[d]), P, phi)
    return obs.assign(**{names[0]: hist_x, names[1]: dyn})


# --- защита от грубой ошибки расчётной кривой ------------------------------------------

def traded_pairs(board: pd.DataFrame) -> set:
    """Пары (день, серия), в которых были сделки по опционам вне денег."""
    return set(board[board["vol_otm"] > 0].groupby(["date", "expiry"]).groups)


def _on_grid(series: tuple) -> np.ndarray:
    """Кривая серии (T, k, iv) в узлах X_GRID по x; NaN там, где кривая не задана."""
    T, ks, iv = series
    k = X_GRID * np.sqrt(T)
    out = np.full(len(X_GRID), np.nan)
    ok = (k >= ks.min()) & (k <= ks.max())
    out[ok] = np.interp(k[ok], ks, iv)
    return out


def guard_curves(smiles: dict, traded: set, p: dict) -> tuple[dict, pd.DataFrame]:
    """Наборы кривых по дням без серий с грубой ошибкой и список убранных серий.

    Проверяются только серии, по которым в этот день не было сделок: кривую такой серии биржа строит
    без опоры на рынок. Опорой служат серии со сделками: кривая сравнивается с интерполяцией двух
    ближайших таких серий по sqrt(T), а если серия со сделками есть только с одной стороны — с ней.
    Если медиана расхождения по узлам X_GRID больше доли GUARD_GAP от уровня опорной кривой, серия
    убирается из набора кривых дня. Тогда при переносе её заменяет ближайшая по сроку серия в
    координате x (см. ``transfer``). В дни, когда ничего не убрано, возвращается тот же объект,
    что в ``smiles``.
    """
    out, rows = {}, []
    for d, sm in smiles.items():
        out[d] = sm
        ser = sorted(sm.series.items(), key=lambda kv: kv[1][0])
        anchors = [i for i, (e, _) in enumerate(ser) if (d, e) in traded]
        if not anchors:
            continue
        curves = {i: _on_grid(ser[i][1]) for i in anchors}
        sqrtT = [np.sqrt(s[0]) for _, s in ser]
        bad = {}
        for i, (e, s) in enumerate(ser):
            if i in curves:
                continue
            lo, hi = [j for j in anchors if j < i], [j for j in anchors if j > i]
            if lo and hi:
                w = (sqrtT[i] - sqrtT[lo[-1]]) / (sqrtT[hi[0]] - sqrtT[lo[-1]])
                ref = (1 - w) * curves[lo[-1]] + w * curves[hi[0]]
            else:
                ref = curves[lo[-1]] if lo else curves[hi[0]]
            gap = _on_grid(s) - ref
            gap = gap[np.isfinite(gap)]
            if len(gap) >= 5 and abs(np.median(gap)) > p["GUARD_GAP"] * np.nanmedian(ref):
                bad[e] = {"T_days": 365 * s[0], "gap_vp": float(100 * np.median(gap)),
                          "ref_vp": float(100 * np.nanmedian(ref))}
        if bad:
            out[d] = copy.copy(sm)
            out[d].series = {e: s for e, s in sm.series.items() if e not in bad}
            rows += [{"date": d, "expiry": e, **v} for e, v in bad.items()]
    return out, pd.DataFrame(rows, columns=["date", "expiry", "T_days", "gap_vp", "ref_vp"])


def guard_effect(obs: pd.DataFrame, removed: pd.DataFrame, prev: dict) -> pd.DataFrame:
    """Что изменила защита, по частям данных: сколько серий убрано из истории, сколько сделок следующего
    дня это затронуло и какова ошибка начального приближения на этих сделках без защиты и с ней."""
    days = obs.groupby("part")["date"].unique()
    rows = []
    for part in days.index:
        used = removed[removed["date"].isin([prev[d] for d in days[part] if d in prev])]   # вчерашние кривые этой части
        o = obs[(obs["part"] == part) & ((obs["dyn_g"] - obs["dyn"]).abs() > 1e-9)]
        err = lambda col: float(100 * np.sqrt(((o[col] - o["iv"]) ** 2).mean())) if len(o) else float("nan")  # noqa: E731
        rows.append({"part": part, "removed": len(used), "trades": len(o), "rmse_raw": err("dyn"), "rmse_guard": err("dyn_g")})
    return pd.DataFrame(rows)


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


def level_target(L: pd.DataFrame, cal: dict, h: int, train_end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Изменение уровня серии за h дней и признак «строка годится для обучения» (день прогноза не
    позже конца обучающего периода)."""
    pos, dates = cal["pos"], cal["dates"]
    target = L["date"].map(lambda d: dates[pos[d] + h] if pos[d] + h < len(dates) else pd.NaT)
    future = L.set_index(["date", "expiry"])["lvl"]
    y = future.reindex(pd.MultiIndex.from_arrays([target, L["expiry"]])).to_numpy() - L["lvl"].to_numpy()
    return y, np.isfinite(y) & np.asarray(target <= train_end)


def gbm_shift(L: pd.DataFrame, cal: dict, h: int, train_end: pd.Timestamp, params: dict) -> pd.Series:
    """Прогноз изменения уровня серии за h дней для каждой пары (день, серия): градиентный бустинг."""
    y, fit = level_target(L, cal, h, train_end)
    model = HistGradientBoostingRegressor(random_state=0, **params).fit(L.loc[fit, FEATURES], y[fit])
    return pd.Series(model.predict(L[FEATURES]), index=pd.MultiIndex.from_frame(L[["date", "expiry"]]))


def nn_shift(L: pd.DataFrame, cal: dict, h: int, train_end: pd.Timestamp, params: dict) -> tuple[pd.Series, int]:
    """То же, что ``gbm_shift``, но нейросетью; возвращает прогноз и выбранное число эпох.

    Сеть: два скрытых слоя, признаки нормированы по обучающим строкам, пропуски заменены нулём.
    Число эпох выбирается по последним ``val_days`` дням обучающего периода: сети учатся на более
    ранних днях, и берётся число эпох с наименьшей ошибкой на этих днях. Ноль эпох означает, что
    любая обученная сеть хуже нулевого прогноза; тогда прогноз изменения равен нулю. С выбранным
    числом эпох сети обучаются заново на всём обучающем периоде, прогноз — среднее по сетям.
    """
    y, fit = level_target(L, cal, h, train_end)
    X = L[FEATURES].to_numpy(float)
    Z = np.nan_to_num((X - np.nanmean(X[fit], axis=0)) / (np.nanstd(X[fit], axis=0) + 1e-12))
    scale = float(y[fit].std())
    torch.set_num_threads(1)
    Zt, yt = torch.tensor(Z, dtype=torch.float32), torch.tensor(np.nan_to_num(y) / scale, dtype=torch.float32)
    val = fit & np.asarray(L["date"] > train_end - pd.Timedelta(days=params["val_days"]))
    index = pd.MultiIndex.from_frame(L[["date", "expiry"]])

    def train(rows: np.ndarray, epochs: int, seed: int, watch: np.ndarray | None = None):
        torch.manual_seed(seed)
        width = params["width"]
        net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), width), torch.nn.GELU(),
                                  torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=params["lr"], weight_decay=params["decay"])
        rows_t, losses = torch.tensor(rows), []
        for epoch in range(epochs):
            opt.zero_grad()
            torch.nn.functional.huber_loss(net(Zt[rows_t]).squeeze(-1), yt[rows_t]).backward()
            opt.step()
            if watch is not None and (epoch + 1) % params["check"] == 0:
                with torch.no_grad():
                    losses.append(float(((net(Zt[torch.tensor(watch)]).squeeze(-1) - yt[torch.tensor(watch)]) ** 2).mean()))
        return net, losses

    seeds = range(params["seeds"])
    curve = np.mean([train(fit & ~val, params["max_epochs"], seed, watch=val)[1] for seed in seeds], axis=0)
    zero = float((yt[torch.tensor(val)] ** 2).mean())            # ошибка нулевого прогноза на тех же днях
    epochs = int(np.argmin([zero, *curve])) * params["check"]
    if epochs == 0:
        return pd.Series(0.0, index=index), 0
    nets = [train(fit, epochs, seed)[0] for seed in seeds]
    with torch.no_grad():
        preds = [net(Zt).squeeze(-1).numpy() * scale for net in nets]
    return pd.Series(np.mean(preds, axis=0), index=index), epochs


def fit_apply(fr: pd.DataFrame, smiles: dict, cal: dict, p: dict) -> tuple[pd.DataFrame, dict]:
    """Оценивает модели на обучающих днях и добавляет прогнозы fc_ar, fc_gbm, fc_carry.

    В параметрах: ``phi`` — доля переноса поправки для каждого горизонта; ``phi_curve`` — её часть,
    которая объясняется сдвигом самой расчётной кривой биржи за h дней; ``ar_beta`` — коэффициенты
    авторегрессии факторов; ``nn_epochs`` — выбранное число эпох нейросети.
    """
    train_end = pd.Timestamp(p["TRAIN_RANGE"][1])
    fac = factor_table(smiles, p)
    dev = fac - fac.rolling(p["FC_AR_WINDOW"], min_periods=p["FC_AR_WINDOW"] // 3).mean()
    L = level_table(smiles, cal, p)
    models = {"phi": {}, "phi_curve": {}, "ar_beta": {}, "nn_epochs": {}}
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
        at = pd.MultiIndex.from_frame(r[["origin", "expiry"]])
        r["fc_gbm"] = r["fc_x"] + gbm_shift(L, cal, h, train_end, p["FC_GBM"]).reindex(at).fillna(0.0).to_numpy()
        nn, models["nn_epochs"][int(h)] = nn_shift(L, cal, h, train_end, p["FC_NN"])
        r["fc_nn"] = r["fc_x"] + nn.reindex(at).fillna(0.0).to_numpy()
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

def run(obs: pd.DataFrame, smiles: dict, prev: dict, cal: dict, p: dict, workers: int, guarded: dict | None = None):
    """Оценивает модели на обучающем периоде и считает прогнозы на остальных днях.

    Возвращает (наблюдения с колонками ``hist_x``, ``dyn`` и ``dyn_g``, прогнозы по сделкам тестового
    и контрольного периодов, параметры моделей). ``resid`` в параметрах — ядро для разности
    «сделка − кривая того же дня», ``phi`` — доля переноса поправки для каждого горизонта.
    ``guarded`` — кривые по дням после ``guard_curves``; по ним считается ``dyn_g``.
    """
    P = fit_resid(obs)
    fr, models = fit_apply(frames(obs, smiles, cal, p["LAGS"], P, workers), smiles, cal, p)
    models["resid"] = P
    out = add_prior(obs, smiles, prev, P, models["phi"][1])
    out = add_prior(out, guarded or smiles, prev, P, models["phi"][1], names=("hist_xg", "dyn_g"))
    out.attrs = dict(obs.attrs)
    return out, fr[fr["part"] != "train"].reset_index(drop=True), models


def save(fr: pd.DataFrame, models: dict, asset: str, p: dict, data_dir, tables_dir,
         guard: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> None:
    """Прогнозы по сделкам (data/), сводные таблицы и параметры моделей ``reports/tables/exp_forecast_*``.
    ``guard`` — список убранных защитой серий и её эффект (``exp_guard_*``)."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    fr.to_csv(data_dir / "processed" / f"exp_forecast_results_{asset}.csv.gz", index=False)
    (tables_dir / f"exp_forecast_params_{asset}.json").write_text(json.dumps(models, ensure_ascii=False, indent=2))
    if guard is not None:
        guard[0].to_csv(tables_dir / f"exp_guard_removed_{asset}.csv", index=False)
        guard[1].to_csv(tables_dir / f"exp_guard_{asset}.csv", index=False)
    for part in ("test", "holdout"):
        r = fr[fr["part"] == part]
        if len(r):
            tag = asset if part == "test" else f"{part}_{asset}"
            rmse_table(r, p["N_BOOT"]).to_csv(tables_dir / f"exp_forecast_{tag}.csv", index=False)
            diff_table(r, p["N_BOOT"]).to_csv(tables_dir / f"exp_forecast_diff_{tag}.csv", index=False)
    bucket_table(fr[fr["part"] == "test"], p["FC_BUCKET_LAGS"]).to_csv(
        tables_dir / f"exp_forecast_buckets_{asset}.csv", index=False)
