"""Проверка по ценам заявок: попадает ли поверхность между лучшей покупкой и лучшей продажей.

Исторических заявок в открытом ISS нет. Есть только срезы текущих заявок, которые собирает
scripts/collect_live_board.py (data/live/{asset}/YYYY-MM-DD_HHMMSS.csv.gz). Срез относится к
моменту времени, поэтому проверяется такой вопрос: поверхность построена по данным торгового
дня t (он закончился до среза), лежит ли она внутри спреда заявок, стоящих после конца дня t.

Заявка учитывается, если у опциона вне денег есть и покупка, и продажа, срок и денежность
в тех же границах, что у сделок эксперимента. Цены заявок переводятся в IV по середине спреда
фьючерса из того же среза. Поверхность дня t читается при той же нормированной денежности x.

Методы:
  exchange — расчётная кривая биржи дня t;
  ssvi     — SSVI по всем сделкам дня t;
  carry    — кривая биржи дня t + перенос поправки по сделкам дня t (прогноз на следующий день);
  gp_dyn   — предлагаемый метод (динамический вариант) по всем сделкам дня t.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import black76, forecast, gp, ssvi

EXPIRY_TIME = pd.Timedelta(hours=18, minutes=50)   # вечерний клиринг: конец торгового дня и момент экспирации
YEAR = pd.Timedelta(days=365)
METHODS = ["exchange", "ssvi", "carry", "gp_dyn"]


def load_snapshot(path: Path, p: dict) -> pd.DataFrame:
    """Двусторонние заявки по опционам вне денег из одного среза: IV покупки и продажи."""
    d = pd.read_csv(path)
    ts = pd.to_datetime(path.name[:17], format="%Y-%m-%d_%H%M%S")
    d = d[(d["BID"] > 0) & (d["OFFER"] > d["BID"]) & (d["UND_BID"] > 0) & (d["UND_OFFER"] > 0)].copy()
    F = 0.5 * (d["UND_BID"] + d["UND_OFFER"]).to_numpy(float)
    K = d["STRIKE"].to_numpy(float)
    expiry = pd.to_datetime(d["LASTTRADEDATE"])
    T = ((expiry + EXPIRY_TIME - ts) / YEAR).to_numpy(float)
    call = (d["OPTIONTYPE"] == "C").to_numpy()
    q = pd.DataFrame({"time": ts, "expiry": expiry.to_numpy(), "strike": K, "T": T, "k": np.log(K / F),
                      "bid": black76.implied_vol(d["BID"].to_numpy(float), F, K, T, call),
                      "ask": black76.implied_vol(d["OFFER"].to_numpy(float), F, K, T, call)})
    q["x"], q["sqrtT"] = q["k"] / np.sqrt(q["T"]), np.sqrt(q["T"])
    otm = np.where(call, K >= F, K < F)
    days = 365 * q["T"]
    ok = (otm & (days >= p["T_MIN_DAYS"]) & (days <= p["T_MAX_DAYS"]) & (q["x"].abs() <= p["X_MAX"])
          & q["bid"].notna() & q["ask"].notna())
    return q[ok].reset_index(drop=True)


def surfaces(q: pd.DataFrame, t, obs_t: pd.DataFrame, context: tuple, vis_prev, p: dict, models: dict) -> pd.DataFrame:
    """IV методов в точках заявок ``q``; все методы используют только данные не позже дня ``t``."""
    smiles, prev, _ = context
    sm, fc = smiles[t], models["fc"]
    q = q.copy()
    q["exchange"], q["carry"] = forecast.dyn_prior(q, sm, obs_t, fc["resid"], fc["phi"][1])
    # Те же точки в координатах дня t: та же x, срок серии в день t.
    T_t = np.array([sm.series[e][0] if e in sm.series else T for e, T in zip(q["expiry"], q["T"])])
    pts = q[["expiry", "x"]].assign(T=T_t, sqrtT=np.sqrt(T_t), k=q["x"] * np.sqrt(T_t))
    q["ssvi"] = ssvi.cb_ssvi(obs_t, pts, p["SSVI_RIDGE"], p["SSVI_FSCALE"])
    pts["hist_x"], pts["dyn"] = forecast.dyn_prior(pts, smiles[prev[t]], vis_prev, fc["resid"], fc["phi"][1])
    q["gp_dyn"] = gp.predict(obs_t, pts, models["gp_dyn"], prior="dyn")
    return q


def summary(q: pd.DataFrame, max_spread: float) -> pd.DataFrame:
    """Доля заявок, внутри спреда которых лежит поверхность, и среднее отклонение от середины спреда."""
    mid, spread = 0.5 * (q["bid"] + q["ask"]), q["ask"] - q["bid"]
    rows = []
    for subset, sel in (("all", np.ones(len(q), bool)), ("narrow", (spread <= max_spread).to_numpy())):
        for m in METHODS:
            inside = (q[m] >= q["bid"]) & (q[m] <= q["ask"])
            rows.append({"subset": subset, "method": m, "n": int(sel.sum()), "series": int(q.loc[sel, "expiry"].nunique()),
                         "spread_median_vp": float(100 * spread[sel].median()),
                         "inside": float(inside[sel].mean()), "mad_vp": float(100 * (q[m] - mid)[sel].abs().mean())})
    return pd.DataFrame(rows)


def run(obs: pd.DataFrame, context: tuple, p: dict, models: dict, live_dir: Path, asset: str, tables_dir: Path):
    """Проверка по всем срезам из ``live_dir``. Пишет ``exp_quotes_{asset}.csv`` или ничего, если срезов нет."""
    smiles, prev, _ = context
    by_day = dict(tuple(obs.groupby("date")))
    parts = []
    for path in sorted(Path(live_dir).glob("*.csv.gz")):
        q = load_snapshot(path, p)
        done = [d for d in smiles if d + EXPIRY_TIME <= q["time"].iloc[0]] if len(q) else []
        t = max(done) if done else None
        if t is not None and t in by_day and t in prev:
            parts.append(surfaces(q, t, by_day[t], context, by_day.get(prev[t]), p, models).assign(day=t))
    if not parts:
        return None
    q = pd.concat(parts, ignore_index=True)
    out = summary(q, p["QUOTES_MAX_SPREAD"]).assign(snapshots=len(parts), day=str(q["day"].max().date()),
                                                    time=str(q["time"].max()))
    out.to_csv(tables_dir / f"exp_quotes_{asset}.csv", index=False)
    return out
