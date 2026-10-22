"""Простые методы восстановления IV в скрытых точках для пилотного эксперимента.

Координаты: k = ln(K/F), x = k / sqrt(T). Все методы возвращают IV в долях единицы.

* ``hist``     — вчерашняя кривая MOEX той же серии (sticky moneyness), без текущих данных;
* ``cb_quad``  — только текущая доска: квадратичная улыбка по x в каждой серии (ridge),
                 для серий без видимых точек — общая поверхность дня по (x, sqrt T);
* ``cb_svi``   — только текущая доска: raw SVI по серии при >= 5 видимых точках, иначе cb_quad;
* ``hist_cb``  — вчерашняя кривая как априор + поправка уровня/наклона по видимым точкам
                 (байесовская ridge-регрессия остатков с априорными масштабами).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

RIDGE = 1e-4  # значения по умолчанию; эксперимент передаёт их из config.py
# Априорные масштабы поправок hist_cb (в долях IV) при шуме наблюдения 1 п.в.
NOISE, TAU_LEVEL, TAU_SLOPE, TAU_SERIES = 0.01, 0.02, 0.02, 0.01


def _ridge(X: np.ndarray, y: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    A = X.T @ X + np.diag(penalty)
    return np.linalg.solve(A, X.T @ y)


class Smiles:
    """Набор кривых одного дня: серия -> (T, k, iv) для интерполяции."""

    def __init__(self, day):
        self.series = {}
        for e, g in day.groupby("expiry"):
            g = g.sort_values("k")
            if len(g) >= 2:
                self.series[e] = (float(g["T"].iloc[0]), g["k"].to_numpy(), g["iv_settle"].to_numpy())

    def __call__(self, expiry, k: np.ndarray, T: float) -> np.ndarray:
        if expiry in self.series:
            _, ks, iv = self.series[expiry]
            return np.interp(k, ks, iv)
        if not self.series:
            return np.full(len(k), np.nan)
        # Новая серия: ближайшая по сроку, в координатах x = k / sqrt(T).
        e = min(self.series, key=lambda s: abs(self.series[s][0] - T))
        Ts, ks, iv = self.series[e]
        return np.interp(k / np.sqrt(T), ks / np.sqrt(Ts), iv)


def cb_quad(vis, hid, ridge: float = RIDGE) -> np.ndarray:
    """vis/hid — DataFrame с колонками expiry, x, sqrtT (+ iv у vis)."""
    out = np.full(len(hid), np.nan)
    Xg = np.column_stack([np.ones(len(vis)), vis["x"], vis["x"] ** 2, vis["sqrtT"], vis["x"] * vis["sqrtT"]])
    beta_g = _ridge(Xg, vis["iv"].to_numpy(), np.array([0, ridge, ridge, ridge, ridge]))
    for i, (e, g) in enumerate(hid.groupby("expiry", sort=False)):
        idx = hid.index.get_indexer(g.index)
        v = vis[vis["expiry"] == e]
        if len(v) >= 1:
            X = np.column_stack([np.ones(len(v)), v["x"], v["x"] ** 2])
            beta = _ridge(X, v["iv"].to_numpy(), np.array([0, ridge, ridge]))
            out[idx] = beta[0] + beta[1] * g["x"] + beta[2] * g["x"] ** 2
        else:
            out[idx] = (np.column_stack([np.ones(len(g)), g["x"], g["x"] ** 2, g["sqrtT"],
                                         g["x"] * g["sqrtT"]]) @ beta_g)
    return out


def svi_fit(k: np.ndarray, iv: np.ndarray, T: float):
    """Raw SVI для полной дисперсии с ограничениями против дикой экстраполяции по 5–10 точкам:
    наклон крыльев b(1 + |rho|) <= 2 (граница Ли), вершина m внутри диапазона данных,
    ширина s не меньше 1/10 диапазона страйков."""
    w = iv**2 * T
    span = max(np.ptp(k), 1e-3)

    def unpack(p):
        a, u, rho, m, s = p
        return a, 2.0 * u / (1.0 + abs(rho)), rho, m, s

    def model(p, kk):
        a, b, rho, m, s = unpack(p)
        return a + b * (rho * (kk - m) + np.sqrt((kk - m) ** 2 + s**2))

    def resid(p):
        return np.sqrt(np.maximum(model(p, k), 1e-10) / T) - iv

    atm_w = float(np.interp(0.0, np.sort(k), w[np.argsort(k)]))
    lower = [-1.0, 0.0, -0.999, k.min(), span / 10]
    upper = [1.0, 1.0, 0.999, k.max(), 2 * span]
    best = None
    for rho0 in (-0.3, 0.0, 0.3):
        u0 = min(0.5, 0.5 * atm_w / span)
        p0 = [0.5 * atm_w, u0, rho0, float(np.clip(0.0, k.min(), k.max())), 0.5 * span]
        res = least_squares(resid, p0, bounds=(lower, upper), max_nfev=2000)
        if best is None or res.cost < best.cost:
            best = res
    return lambda kk: np.sqrt(np.maximum(model(best.x, kk), 1e-10) / T)


def cb_svi(vis, hid, ridge: float = RIDGE) -> np.ndarray:
    out = cb_quad(vis, hid, ridge)
    for e, g in hid.groupby("expiry", sort=False):
        v = vis[vis["expiry"] == e]
        if len(v) >= 5:
            f = svi_fit(v["k"].to_numpy(), v["iv"].to_numpy(), float(v["T"].iloc[0]))
            out[hid.index.get_indexer(g.index)] = f(g["k"].to_numpy())
    return out


def hist_cb(vis, hid, prior: tuple[float, float, float, float] = (NOISE, TAU_LEVEL, TAU_SLOPE, TAU_SERIES)
            ) -> np.ndarray:
    """vis/hid должны содержать колонку ``hist`` — прогноз вчерашней кривой."""
    series = sorted(set(vis["expiry"]) | set(hid["expiry"]))
    col = {e: i for i, e in enumerate(series)}

    def design(d):
        X = np.zeros((len(d), 2 + len(series)))
        X[:, 0] = 1.0
        X[:, 1] = d["x"]
        X[np.arange(len(d)), 2 + d["expiry"].map(col).to_numpy()] = 1.0
        return X

    noise, tau_level, tau_slope, tau_series = prior
    penalty = noise**2 / np.array([tau_level**2, tau_slope**2] + [tau_series**2] * len(series))
    beta = _ridge(design(vis), (vis["iv"] - vis["hist"]).to_numpy(), penalty)
    return hid["hist"].to_numpy() + design(hid) @ beta
