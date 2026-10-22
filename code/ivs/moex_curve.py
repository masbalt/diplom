"""Кривая волатильности НКЦ, по которой MOEX считает теоретические и расчётные цены опционов.

«Порядок расчёта кривых волатильности» (НКЦ), раздел III:

    sigma = a + b * (1 - exp(-c * y^2)) + d * arctg(e * y) / e,   (в процентах)
    x = ln(Strike / F) / sqrt(T),   y = x - s / sqrt(T).

Параметры подстраиваются по лучшим bid/ask каждой серии отдельно, начиная с предыдущих значений.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def smile(x, T, s, a, b, c, d, e):
    y = np.asarray(x) - s / np.sqrt(T)
    return a + b * (1.0 - np.exp(-c * y**2)) + d * np.arctan(e * y) / e


def fit(x, T, iv_pct, weights=None):
    """МНК-подгонка 6 параметров к IV одной серии (в процентах). Возвращает (params, rmse)."""
    x, iv_pct = np.asarray(x, float), np.asarray(iv_pct, float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    atm = float(np.interp(0.0, np.sort(x), iv_pct[np.argsort(x)]))
    best = None
    for c0, e0 in [(1.0, 1.0), (5.0, 3.0), (0.3, 0.5)]:
        p0 = [0.0, atm, 1.0, c0, -1.0, e0]

        def resid(p):
            return np.sqrt(w) * (smile(x, T, *p) - iv_pct)

        res = least_squares(
            resid, p0,
            bounds=([-5, 0, -200, 1e-6, -200, 1e-6], [5, 500, 500, 100, 200, 100]),
            max_nfev=4000,
        )
        if best is None or res.cost < best.cost:
            best = res
    rmse = float(np.sqrt(np.mean((smile(x, T, *best.x) - iv_pct) ** 2)))
    return best.x, rmse
