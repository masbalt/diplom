"""SSVI (Gatheral, Jacquier, 2014): поверхность полной дисперсии без статического арбитража.

    w(k, theta) = theta / 2 * (1 + rho * phi * k + sqrt((phi * k + rho)^2 + 1 - rho^2)),
    phi(theta)  = eta / (theta^gamma * (1 + theta)^(1 - gamma)),

где theta — полная дисперсия серии на деньгах (k = 0). Параметры rho, eta, gamma общие для
всех серий, у каждой серии свой theta. Арбитража нет по построению, если

    * theta не убывает по сроку (нет календарного арбитража),
    * 0 < gamma <= 1/2 и eta * (1 + |rho|) <= 2 (нет арбитража «бабочка»).

Оба условия зашиты в параметризацию, поэтому подгонка идёт без ограничений.
Это сильный классический метод только текущей доски: серии связаны общей формой улыбки.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

GAMMA_MAX = 0.5
THETA_MIN = 1e-8


def total_variance(k, theta, rho: float, eta: float, gamma: float):
    k, theta = np.asarray(k, float), np.asarray(theta, float)
    phi = eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1.0 - rho**2))


def shape(raw) -> tuple[float, float, float]:
    """Свободные параметры -> (rho, eta, gamma), для которых нет арбитража «бабочка»."""
    rho = float(np.tanh(raw[0]))
    eta = float(2.0 * expit(raw[1]) / (1.0 + abs(rho)))
    gamma = float(GAMMA_MAX * expit(raw[2]))
    return rho, eta, gamma


class Surface:
    """Подогнанная поверхность: theta серий без точек интерполируется по сроку."""

    def __init__(self, T: np.ndarray, theta: np.ndarray, rho: float, eta: float, gamma: float):
        self.T, self.theta, self.rho, self.eta, self.gamma = T, theta, rho, eta, gamma

    def theta_at(self, T) -> np.ndarray:
        """Линейная интерполяция theta по сроку; за краями — постоянная волатильность на деньгах.

        Так theta остаётся неубывающей по сроку и для серий, где не было видимых сделок.
        """
        T = np.asarray(T, float)
        out = np.interp(T, self.T, self.theta)
        out = np.where(T < self.T[0], self.theta[0] * T / self.T[0], out)
        return np.where(T > self.T[-1], self.theta[-1] * T / self.T[-1], out)

    def iv(self, k, T) -> np.ndarray:
        T = np.asarray(T, float)
        w = total_variance(k, self.theta_at(T), self.rho, self.eta, self.gamma)
        return np.sqrt(np.maximum(w, 1e-12) / T)


def fit(k, T, iv, ridge: float = 1e-4, f_scale: float | None = 0.01) -> Surface:
    """Подгонка SSVI к точкам (k, T, iv) нескольких серий; серия задаётся значением T.

    ``f_scale`` — масштаб робастной функции потерь soft-L1 в долях IV (None — обычный МНК).
    ``ridge`` слегка притягивает параметры формы к нейтральным значениям: при 5–10 точках на
    всю доску они иначе не определяются.
    """
    k, T, iv = (np.asarray(a, float) for a in (k, T, iv))
    Ts, inv = np.unique(T, return_inverse=True)

    # Начальное theta серии: полная дисперсия ближайшей к деньгам точки, затем монотонизация.
    theta0 = np.empty(len(Ts))
    for i in range(len(Ts)):
        m = inv == i
        j = np.argmin(np.abs(k[m]))
        theta0[i] = iv[m][j] ** 2 * Ts[i]
    theta0 = np.maximum.accumulate(np.maximum(theta0, THETA_MIN))
    steps0 = np.log(np.maximum(np.diff(theta0, prepend=0.0), 0.02 * theta0))

    def unpack(p):
        theta = np.cumsum(np.exp(np.clip(p[3:], -60.0, 5.0)))
        return theta, *shape(p[:3])

    def resid(p):
        theta, rho, eta, gamma = unpack(p)
        w = total_variance(k, theta[inv], rho, eta, gamma)
        r = np.sqrt(np.maximum(w, 1e-12) / T) - iv
        return np.concatenate([r, np.sqrt(ridge) * p[:3]])

    kw = {} if f_scale is None else {"loss": "soft_l1", "f_scale": f_scale}
    best = None
    for rho0 in (0.0, 0.5, -0.5):
        p0 = np.concatenate([[np.arctanh(rho0), 0.0, 0.0], steps0])
        res = least_squares(resid, p0, max_nfev=300, **kw)
        if best is None or res.cost < best.cost:
            best = res
    theta, rho, eta, gamma = unpack(best.x)
    return Surface(Ts, theta, rho, eta, gamma)


def cb_ssvi(vis, hid, ridge: float = 1e-4, f_scale: float | None = 0.01) -> np.ndarray:
    """Метод только текущей доски: SSVI по видимым сделкам, прогноз IV в точках ``hid``."""
    s = fit(vis["k"].to_numpy(), vis["T"].to_numpy(), vis["iv"].to_numpy(), ridge, f_scale)
    return s.iv(hid["k"].to_numpy(), hid["T"].to_numpy())
