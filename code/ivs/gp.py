"""Предлагаемый метод: вчерашняя поверхность как априор + сглаженная поправка по видимым сделкам.

Модель сделки i дня t (серия e_i, срок T_i, денежность x_i = k_i / sqrt(T_i)):

    iv_i = prior_i + delta(x_i, e_i) + eps_i,

    prior_i — вчерашняя поверхность в этой точке (0 для варианта без истории),
    delta   — поправка дня, гауссовский процесс с нулевым средним и ковариацией K,
    eps_i   — шум сделки с дисперсией a^2 + b^2 / n_i (n_i — число минут со сделками).

Ковариация поправки складывается из трёх частей:

    K = [g0^2 + g1^2 xx' + g2^2 (xx')^2]                       общая для всех серий
      + rho(T, T') [t0^2 + t1^2 xx' + t2^2 (xx')^2]            затухает с расстоянием между сроками
      + 1[e = e'] [ts^2 + tloc^2 exp(-(x - x')^2 / (2 ell_x^2))]   внутри одной серии,

    rho(T, T') = exp(-|ln T - ln T'| / ell_T).

Слагаемые 1, xx', (xx')^2 отвечают за сдвиг уровня, наклона и кривизны улыбки. Параметры
(g, t, ts, tloc, ell_T, ell_x, a, b) оцениваются по прошлым дням максимизацией правдоподобия:
так из истории берётся не только вчерашняя поверхность, но и то, как обычно выглядят поправки.

Оценка поправки — апостериорное среднее. Оно же решает задачу гребневой регрессии в ядровом
пространстве: min sum_i (r_i - delta_i)^2 / s_i^2 + ||delta||_K^2, r_i = iv_i - prior_i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

NAMES = ["g0", "g1", "g2", "t0", "t1", "t2", "ell_T", "ts", "tloc", "ell_x", "a", "b"]
OFF = 1e-6  # значение «выключенного» масштаба
# Начальные значения для поправки ко вчерашней кривой (доли IV) и границы поиска.
INIT = {"g0": 0.005, "g1": 0.01, "g2": 0.01, "t0": 0.01, "t1": 0.03, "t2": 0.05, "ell_T": 0.5,
        "ts": 0.005, "tloc": 0.005, "ell_x": 0.1, "a": 0.004, "b": 0.01}
# Начальные значения для варианта без истории: ядро описывает саму поверхность, а не поправку.
INIT_NO_PRIOR = {**INIT, "g0": 0.2, "g1": 0.1, "g2": 0.3, "t0": 0.05, "t1": 0.1, "t2": 0.2}
BOUNDS = {"ell_T": (0.02, 20.0), "ell_x": (0.02, 1.0)}
SCALE_BOUNDS = (OFF, 2.0)


def arrays(d: pd.DataFrame, prior: str | None = "hist") -> dict:
    """Числовые массивы одного набора точек (DataFrame с колонками x, T, expiry и др.)."""
    out = {
        "x": d["x"].to_numpy(float),
        "lnT": np.log(d["T"].to_numpy(float)),
        "e": d["expiry"].to_numpy().astype("datetime64[D]").astype(np.int64),
        "prior": d[prior].to_numpy(float) if prior else np.zeros(len(d)),
    }
    if "iv" in d:
        out["iv"] = d["iv"].to_numpy(float)
    if "n_candles" in d:
        out["n"] = d["n_candles"].to_numpy(float)
    return out


def kernel(A: dict, B: dict, P: dict) -> np.ndarray:
    xa, xb = A["x"][:, None], B["x"][None, :]
    xx = xa * xb
    same = A["e"][:, None] == B["e"][None, :]
    if P.get("indep"):   # вариант без связей между сериями
        rho = same.astype(float)
    else:
        rho = np.exp(-np.abs(A["lnT"][:, None] - B["lnT"][None, :]) / P["ell_T"])
    K = P["g0"] ** 2 + P["g1"] ** 2 * xx + P["g2"] ** 2 * xx**2
    K = K + rho * (P["t0"] ** 2 + P["t1"] ** 2 * xx + P["t2"] ** 2 * xx**2)
    return K + same * (P["ts"] ** 2 + P["tloc"] ** 2 * np.exp(-0.5 * (xa - xb) ** 2 / P["ell_x"] ** 2))


def kernel_diag(A: dict, P: dict) -> np.ndarray:
    x2 = A["x"] ** 2
    return (P["g0"] ** 2 + P["t0"] ** 2 + (P["g1"] ** 2 + P["t1"] ** 2) * x2
            + (P["g2"] ** 2 + P["t2"] ** 2) * x2**2 + P["ts"] ** 2 + P["tloc"] ** 2)


def noise_var(n: np.ndarray, P: dict) -> np.ndarray:
    return P["a"] ** 2 + P["b"] ** 2 / n


# --- оценка параметров по прошлым дням ------------------------------------------

def neg_log_lik(P: dict, days: list[dict]) -> float:
    """Средний на точку отрицательный логарифм правдоподобия остатков r = iv - prior."""
    total, n = 0.0, 0
    for A in days:
        K = kernel(A, A, P) + np.diag(noise_var(A["n"], P))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e6
        r = A["iv"] - A["prior"]
        total += 0.5 * r @ cho_solve(c, r) + np.log(np.diag(c[0])).sum()
        n += len(r)
    return total / n + 0.5 * np.log(2 * np.pi)


def fit(days: list[dict], init: dict = INIT, fixed: dict | None = None, flags: dict | None = None) -> dict:
    """Максимум правдоподобия по списку дней (``arrays``). ``fixed`` — параметры, которые не
    оцениваются (для вариантов метода), ``flags`` — признаки вроде ``indep``."""
    fixed, flags = fixed or {}, flags or {}
    free = [k for k in NAMES if k not in fixed]

    def params(theta):
        return {**dict(zip(free, np.exp(theta))), **fixed, **flags}

    bounds = [tuple(np.log(BOUNDS.get(k, SCALE_BOUNDS))) for k in free]
    theta0 = np.log([init[k] for k in free])
    res = minimize(lambda th: neg_log_lik(params(th), days), theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})
    P = params(res.x)
    P["nll"] = float(res.fun)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in P.items()}


# --- восстановление ---------------------------------------------------------------

def predict(vis: pd.DataFrame, hid: pd.DataFrame, P: dict, prior: str | None = "hist",
            return_sd: bool = False):
    """IV в точках ``hid`` по видимым сделкам ``vis``. С ``return_sd`` — ещё и стандартное
    отклонение восстановленной поверхности в этих точках (без шума самой сделки)."""
    A, B = arrays(vis, prior), arrays(hid, prior)
    c = cho_factor(kernel(A, A, P) + np.diag(noise_var(A["n"], P)), lower=True)
    Ks = kernel(B, A, P)
    mean = B["prior"] + Ks @ cho_solve(c, A["iv"] - A["prior"])
    if not return_sd:
        return mean
    var = kernel_diag(B, P) - np.einsum("ij,ji->i", Ks, cho_solve(c, Ks.T))
    return mean, np.sqrt(np.maximum(var, 0.0))
