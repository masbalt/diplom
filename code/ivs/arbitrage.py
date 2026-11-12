"""Статический арбитраж поверхности на сетке: проверка и исправление.

Поверхность задаётся значениями IV в узлах (серия, k). Всё считается в нормированных ценах
колл-опциона c = C / F как функции страйка m = K / F = exp(k). Арбитража на сетке нет, если

    * «бабочка»: в каждой серии c выпукла по m и не возрастает, (1 - m)+ <= c <= 1;
    * «календарь»: при одинаковом k цена c не убывает по сроку (то же, что неубывание
      полной дисперсии w = iv^2 T).

Все условия записаны в единицах цены, поэтому допуск один — доля шага цены опциона.
Нарушение меньше допуска арбитражем не считается: на нём нельзя заработать.

Все условия линейны по c. Ближайшая безарбитражная поверхность ищется линейным
программированием: минимизируется сумма |c - c0| / vega, то есть примерно сумма модулей
поправок IV (как в Cohen, Reisinger, Wang, 2020). Вега берётся при волатильности серии на
деньгах: собственная вега узла с вырожденной волатильностью почти нулевая, и его сдвиг
казался бы намного дороже, чем он есть. Узлам можно задать веса: чем увереннее
метод в значении поверхности в узле, тем дороже его сдвигать. Без весов исправление может
сдвинуть серию с большим числом сделок вместо соседней серии, о которой почти ничего не известно.

Рабочая зона сетки — стандартизованная денежность |z| <= z_max, z = k / (sigma_ATM sqrt(T)).
Дальше неё цены коротких серий сравнимы с шагом цены, и IV определяется округлением.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack
from scipy.special import ndtr

from . import black76

VEGA_FLOOR = 1e-3   # нижняя граница плотности в веге: не даёт бесконечных весов на крыльях
IV_FLOOR = 1e-3     # IV узла, цена которого после исправления равна внутренней стоимости
SQRT_2PI = np.sqrt(2.0 * np.pi)


def make_grid(series: pd.DataFrame, z_max: float, step: float) -> pd.DataFrame:
    """Узлы сетки. ``series`` — колонки expiry, T, atm (IV на деньгах), по строке на серию.

    Узлы всех серий лежат на общей равномерной сетке по k с шагом ``step``. Серия использует
    узлы с |k| <= z_max * atm * sqrt(T); радиус не убывает по сроку, поэтому каждый узел
    короткой серии есть и у следующей, и календарное условие проверяется в одних и тех же k.
    """
    s = series.sort_values("T").drop_duplicates("expiry").reset_index(drop=True)
    radius = np.maximum.accumulate(z_max * s["atm"].to_numpy() * np.sqrt(s["T"].to_numpy()))
    parts = []
    for e, T, atm, r in zip(s["expiry"], s["T"], s["atm"], radius):
        n = max(int(np.floor(r / step)), 2)
        parts.append(pd.DataFrame({"expiry": e, "T": T, "atm": atm, "j": np.arange(-n, n + 1)}))
    g = pd.concat(parts, ignore_index=True)
    g["k"] = g["j"] * step
    g["sqrtT"] = np.sqrt(g["T"])
    g["x"] = g["k"] / g["sqrtT"]
    return g


def call_price(k, iv, T):
    """Нормированная цена колла C / F и вега (производная по IV)."""
    k, iv, T = (np.asarray(a, float) for a in (k, iv, T))
    sd = iv * np.sqrt(T)
    d1 = -k / sd + 0.5 * sd
    c = ndtr(d1) - np.exp(k) * ndtr(d1 - sd)
    vega = np.sqrt(T) * np.exp(-0.5 * d1**2) / SQRT_2PI
    return c, vega


def constraints(grid: pd.DataFrame):
    """Линейные условия отсутствия арбитража в единицах цены: A c >= lb.

    Возвращает (A, lb, is_calendar). «Бабочка» в узле j: цена не выше линейной интерполяции
    соседей, lam * c[j-1] - c[j] + (1 - lam) * c[j+1] >= 0. Слева от первого узла соседом
    служит страйк 0 с ценой 1.
    """
    rows, cols, vals, lb, cal = [], [], [], [], []

    def add(idx, coef, bound, is_cal):
        r = len(lb)
        rows.extend([r] * len(idx)); cols.extend(idx); vals.extend(coef)
        lb.append(bound); cal.append(is_cal)

    groups = [g.index.to_numpy() for _, g in grid.groupby("T", sort=True)]
    m = np.exp(grid["k"].to_numpy())
    node = grid["j"].to_numpy()
    for idx in groups:
        mm = m[idx]
        d = np.diff(mm)
        lam = d[0] / (mm[0] + d[0])
        add([idx[0], idx[1]], [-1.0, 1.0 - lam], -lam, False)
        for j in range(1, len(idx) - 1):
            lam = d[j] / (d[j - 1] + d[j])
            add([idx[j - 1], idx[j], idx[j + 1]], [lam, -1.0, 1.0 - lam], 0.0, False)
        add([idx[-2], idx[-1]], [1.0, -1.0], 0.0, False)   # последний наклон не положителен
    for near, far in zip(groups[:-1], groups[1:]):
        pos = dict(zip(node[far], far))
        for i in near:
            add([pos[node[i]], i], [1.0, -1.0], 0.0, True)
    A = coo_matrix((vals, (rows, cols)), shape=(len(lb), len(grid))).tocsr()
    return A, np.array(lb), np.array(cal)


def violations(grid: pd.DataFrame, iv: np.ndarray, tol: float) -> dict:
    """Доли нарушенных условий «бабочка» и «календарь» на сетке; ``tol`` — допуск в единицах c."""
    A, lb, cal = constraints(grid)
    c, _ = call_price(grid["k"], iv, grid["T"])
    bad = (A @ c - lb) < -tol
    return {
        "butterfly_share": float(bad[~cal].mean()),
        "calendar_share": float(bad[cal].mean()) if cal.any() else 0.0,
        "butterfly_any": bool(bad[~cal].any()),
        "calendar_any": bool(bad[cal].any()),
    }


def repair(grid: pd.DataFrame, iv: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Ближайшая безарбитражная поверхность на сетке: IV в узлах после исправления.

    ``weight`` — относительная цена сдвига узла (по умолчанию одинаковая для всех узлов).
    """
    iv = np.asarray(iv, float)
    A, lb, _ = constraints(grid)
    k, T = grid["k"].to_numpy(), grid["T"].to_numpy()
    c0, _ = call_price(k, iv, T)
    n = len(c0)
    if ((A @ c0 - lb) >= 0).all():
        return iv.copy()
    _, vega = call_price(k, grid["atm"].to_numpy(), T)   # вега при волатильности серии на деньгах
    w = (1.0 if weight is None else np.asarray(weight, float)) / np.maximum(vega, VEGA_FLOOR * np.sqrt(T))
    W = eye(n, format="csr").multiply(w[:, None]).tocsr()
    I = eye(n, format="csr")
    A_ub = vstack([hstack([-A, coo_matrix((A.shape[0], n))]), hstack([W, -I]), hstack([-W, -I])]).tocsr()
    b_ub = np.concatenate([-lb, W @ c0, -(W @ c0)])
    intrinsic = np.maximum(1.0 - np.exp(k), 0.0)
    bounds = [(lo, 1.0) for lo in intrinsic] + [(0.0, None)] * n
    res = linprog(np.concatenate([np.zeros(n), np.ones(n)]), A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  method="highs")
    if res.status != 0:
        raise RuntimeError(f"arbitrage repair failed: {res.message}")
    c = res.x[:n]
    out = iv.copy()
    moved = np.abs(c - c0) > 1e-10
    if moved.any():
        is_call = k[moved] >= 0
        price = np.where(is_call, c[moved], c[moved] + np.exp(k[moved]) - 1.0)   # OTM-сторона
        new = black76.implied_vol(price, 1.0, np.exp(k[moved]), T[moved], is_call)
        # Цена на границе (нулевая временная стоимость) не обращается в IV: берётся минимальная IV.
        out[moved] = np.where(np.isfinite(new), new, IV_FLOOR)
    return out


def shift_points(grid: pd.DataFrame, d_iv: np.ndarray, pts: pd.DataFrame) -> np.ndarray:
    """Поправка IV в точках ``pts`` (expiry, k): линейная интерполяция по k внутри серии,
    ноль за пределами узлов серии."""
    out = np.zeros(len(pts))
    d = pd.Series(d_iv, index=grid.index)
    for e, g in pts.groupby("expiry", sort=False):
        nodes = grid[grid["expiry"] == e]
        if len(nodes):
            out[pts.index.get_indexer(g.index)] = np.interp(g["k"], nodes["k"], d[nodes.index], left=0.0, right=0.0)
    return out
