"""Сборка дневной опционной доски: одна строка на (серия, страйк) в OTM-соглашении.

* F серии восстанавливается по паритету колл–пут на расчётных ценах (``C - P = F - K``
  для маржируемых опционов) — медиана по ближайшим к ATM страйкам;
* ``iv_settle`` — IV расчётной цены MOEX (плотная модельная кривая НКЦ);
* ``iv_trade`` — IV средневзвешенной цены сделок дня по OTM-опциону (реальные наблюдения);
* ``*_err`` — неопределённость IV из-за округления цены до шага: 0.5 * tick / vega.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import black76
from .contracts import parse_secids

PRICE_COLS = {
    "SETTLEPRICE": "settle",
    "THEOR_PRICE": "theor",
    "WAPRICE": "wap",
    "CLOSE": "close",
    "VOLUME": "vol",
    "NUMTRADES": "ntr",
    "OPENPOSITION": "oi",
}
N_PARITY = 5


def _parity_forward(g: pd.DataFrame) -> float:
    both = g[(g["settle_c"] > 0) & (g["settle_p"] > 0)]
    if both.empty:
        return np.nan
    near = both.assign(gap=(both["settle_c"] - both["settle_p"]).abs()).nsmallest(N_PARITY, "gap")
    return float(np.median(near["settle_c"] - near["settle_p"] + near["strike"]))


def build_board(raw: pd.DataFrame, lookup: pd.DataFrame, tick: float) -> pd.DataFrame:
    """``raw`` — итоги дня по опционам одного БА (ISS history), ``lookup`` — справочник серий."""
    if raw.empty:
        return pd.DataFrame()
    df = raw[["TRADEDATE", "SECID", *PRICE_COLS]].rename(columns=PRICE_COLS)
    df = df.join(parse_secids(df["SECID"])).dropna(subset=["group", "strike"])
    df = df.join(lookup, on="group").dropna(subset=["expiry"])
    # Только маржируемые опционы на фьючерсы (SHORTNAME вида «Si-12.26M…»): премиальные
    # европейские опционы (например, Si74.5CL6) имеют другие единицы страйка.
    df = df[df["underlying"].fillna("") != ""]
    date = pd.Timestamp(df["TRADEDATE"].iloc[0])

    value_cols = list(PRICE_COLS.values())
    key = ["expiry", "strike"]
    df = df.rename(columns={"SECID": "secid"})
    calls = df[df["opt_type"] == "C"].groupby(key)[value_cols + ["secid"]].first()
    puts = df[df["opt_type"] == "P"].groupby(key)[value_cols + ["secid"]].first()
    b = calls.join(puts, how="outer", lsuffix="_c", rsuffix="_p").reset_index()
    for col in value_cols:
        b[[f"{col}_c", f"{col}_p"]] = b[[f"{col}_c", f"{col}_p"]].fillna(0.0)

    underlying = df.groupby("expiry")["underlying"].first()
    b["underlying"] = b["expiry"].map(underlying)
    b["date"] = date
    b["T"] = (b["expiry"] - date).dt.days / 365.0
    b = b[b["T"] > 0].copy()
    if b.empty:
        return b
    b["F"] = b["expiry"].map(b.groupby("expiry").apply(_parity_forward, include_groups=False))
    b = b.dropna(subset=["F"])

    F, K, T = b["F"].to_numpy(), b["strike"].to_numpy(), b["T"].to_numpy()
    otm_call = K >= F
    b["k"] = np.log(K / F)

    settle_otm = np.where(otm_call, b["settle_c"], b["settle_p"])
    b["iv_settle"] = black76.implied_vol(settle_otm, F, K, T, otm_call)
    b["iv_settle_err"] = 0.5 * tick / black76.vega(F, K, T, b["iv_settle"])

    # Сделки по OTM-стороне: WAPRICE, при отсутствии — цена последней сделки.
    vol_otm = np.where(otm_call, b["vol_c"], b["vol_p"])
    wap = np.where(otm_call, b["wap_c"], b["wap_p"])
    close = np.where(otm_call, b["close_c"], b["close_p"])
    trade_px = np.where(wap > 0, wap, close)
    trade_px = np.where(vol_otm > 0, trade_px, np.nan)
    b["secid_otm"] = np.where(otm_call, b["secid_c"], b["secid_p"])
    b["vol_otm"] = vol_otm
    b["ntr_otm"] = np.where(otm_call, b["ntr_c"], b["ntr_p"])
    b["price_trade"] = trade_px
    b["iv_trade"] = black76.implied_vol(trade_px, F, K, T, otm_call)
    b["iv_trade_err"] = 0.5 * tick / black76.vega(F, K, T, b["iv_trade"])

    # Сделки по ITM-стороне, пересчитанные паритетом в OTM-эквивалент (менее надёжны).
    vol_itm = np.where(otm_call, b["vol_p"], b["vol_c"])
    wap_itm = np.where(otm_call, b["wap_p"], b["wap_c"])
    itm_px = np.where(vol_itm > 0, wap_itm - np.abs(F - K), np.nan)
    b["vol_itm"] = vol_itm
    b["iv_trade_itm"] = black76.implied_vol(itm_px, F, K, T, otm_call)

    # Стандартизованная денежность z = k / (sigma_ATM sqrt(T)).
    def atm(g):
        ok = g["iv_settle"].notna()
        if ok.sum() < 2:
            return np.nan
        g = g[ok].sort_values("k")
        return float(np.interp(0.0, g["k"], g["iv_settle"]))

    b["atm_iv"] = b["expiry"].map(b.groupby("expiry").apply(atm, include_groups=False))
    b["z"] = b["k"] / (b["atm_iv"] * np.sqrt(b["T"]))
    return b.reset_index(drop=True)
