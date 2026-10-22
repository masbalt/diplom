"""Модель Блэка для маржируемых опционов на фьючерсы MOEX (без дисконтирования).

Маржируемые опционы MOEX оцениваются как ``C = F N(d1) - K N(d2)``, ``P = C + K - F``
(см. «Методику расчёта теоретической цены опциона», п. 5), т. е. Black-76 при r = 0.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

SQRT_2PI = np.sqrt(2.0 * np.pi)


def price(F, K, T, sigma, is_call):
    F, K, T, sigma = (np.asarray(a, dtype=float) for a in (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    call = F * ndtr(d1) - K * ndtr(d2)
    return np.where(is_call, call, call + K - F)


def vega(F, K, T, sigma):
    """dPrice/dSigma (одинакова для колла и пута)."""
    F, K, T, sigma = (np.asarray(a, dtype=float) for a in (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / sd + 0.5 * sd
    return F * np.sqrt(T) * np.exp(-0.5 * d1**2) / SQRT_2PI


def implied_vol(p, F, K, T, is_call, lo: float = 1e-4, hi: float = 10.0, n_iter: int = 80):
    """Векторная подразумеваемая волатильность бисекцией (цена монотонна по sigma).

    Возвращает NaN, если цена вне безарбитражных границ ``(intrinsic, upper)``.
    """
    p, F, K, T = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (p, F, K, T)))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), p.shape)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    upper = np.where(is_call, F, K)
    ok = np.isfinite(p) & np.isfinite(F) & (T > 0) & (p > intrinsic) & (p < upper) & (F > 0) & (K > 0)

    a = np.full(p.shape, lo)
    b = np.full(p.shape, hi)
    for _ in range(n_iter):
        mid = 0.5 * (a + b)
        too_low = price(F, K, T, mid, is_call) < p
        a = np.where(too_low, mid, a)
        b = np.where(too_low, b, mid)
    iv = 0.5 * (a + b)
    iv[~ok] = np.nan
    # Решение упёрлось в границу интервала поиска — считаем ненадёжным.
    iv[(iv <= lo * 1.01) | (iv >= hi * 0.99)] = np.nan
    return iv
