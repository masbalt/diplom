"""IV сделок, синхронизированная с ценой фьючерса по минутным свечам ISS.

Дневная VWAP опциона и расчётная цена фьючерса относятся к разным моментам времени:
при внутридневном ходе фьючерса на 1–2 % IV около ATM искажается на несколько
пунктов волатильности. Поэтому IV считается по каждой минутной свече опциона
с ценой закрытия свечи фьючерса той же минуты и затем усредняется с весами объёма.

Торговый день MOEX D включает вечернюю сессию предыдущего торгового дня (с 19:00),
поэтому свечи берутся в окне [D_prev 19:00, D 19:00).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import black76
from .data import DEFAULT_TICK, client_for
from .iss import ISSClient

EXPIRY_TIME = pd.Timedelta(hours=18, minutes=50)
YEAR = pd.Timedelta(days=365)
SESSION_SPLIT = pd.Timedelta(hours=19)


def candles(client: ISSClient, market: str, secid: str, start: str, end: str) -> pd.DataFrame:
    c = client.table(
        f"engines/futures/markets/{market}/securities/{secid}/candles",
        "candles",
        {"from": start, "till": end, "interval": 1},
    )
    if c.empty:
        return c
    c["begin"] = pd.to_datetime(c["begin"])
    return c[c["volume"] > 0]


def trading_window(date: pd.Timestamp, prev_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    return prev_date + SESSION_SPLIT, date + SESSION_SPLIT


def synced_trade_iv(
    client: ISSClient,
    board_day: pd.DataFrame,
    futures_secid: dict[str, str],
    prev_date: pd.Timestamp,
    tick: float,
) -> pd.DataFrame:
    """Для каждой торговавшейся OTM-строки доски дня — IV, синхронизированная по минутам.

    ``board_day`` — строки ``ivs.board`` одного дня с ``vol_otm > 0`` и колонкой ``secid_otm``;
    ``futures_secid`` — соответствие SHORTNAME фьючерса (``Si-12.26``) его SECID (``SiZ6``).
    """
    date = board_day["date"].iloc[0]
    lo, hi = trading_window(date, prev_date)
    start, end = lo.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")

    fut_px: dict[str, pd.Series] = {}
    for und in board_day["underlying"].unique():
        sec = futures_secid.get(und)
        if sec is None:
            continue
        fc = candles(client, "forts", sec, start, end)
        if not fc.empty:
            fc = fc[(fc["begin"] >= lo) & (fc["begin"] < hi)]
            fut_px[und] = fc.set_index("begin")["close"].sort_index()

    rows = []
    for r in board_day.itertuples(index=False):
        f = fut_px.get(r.underlying)
        if f is None or f.empty:
            continue
        oc = candles(client, "options", r.secid_otm, start, end)
        if oc.empty:
            continue
        oc = oc[(oc["begin"] >= lo) & (oc["begin"] < hi)]
        if oc.empty:
            continue
        # Последняя цена фьючерса не позже конца минутной свечи опциона.
        F = f.reindex(f.index.union(oc["begin"])).ffill().reindex(oc["begin"]).to_numpy()
        T = ((r.expiry + EXPIRY_TIME - oc["begin"]) / YEAR).to_numpy()
        px = oc["close"].to_numpy(float)
        is_call = np.full(len(px), r.strike >= r.F)
        iv = black76.implied_vol(px, F, r.strike, T, is_call)
        err = 0.5 * tick / black76.vega(F, r.strike, T, iv)
        w = oc["volume"].to_numpy(float)
        ok = np.isfinite(iv) & np.isfinite(F) & (err < 0.01)
        if not ok.any():
            continue
        w, iv, F = w[ok], iv[ok], F[ok]
        rows.append({
            "date": date,
            "expiry": r.expiry,
            "strike": r.strike,
            "iv_sync": float(np.average(iv, weights=w)),
            "iv_sync_std": float(np.sqrt(np.average((iv - np.average(iv, weights=w)) ** 2, weights=w))),
            "k_sync": float(np.average(np.log(r.strike / F), weights=w)),
            "vol_sync": float(w.sum()),
            "n_candles": int(ok.sum()),
        })
    return pd.DataFrame(rows)


def sync_asset(root: Path, asset: str, start: str, end: str, workers: int, mode: str = "daily") -> pd.DataFrame:
    """Синхронизированные IV сделок за [start, end]: читает доску ``board_{mode}_{asset}`` и фьючерсы,
    пишет ``processed/trades_sync_{asset}.csv.gz`` (для mode != "daily" — ``trades_sync_{mode}_{asset}``)."""
    root = Path(root)
    client = client_for(root)
    board = pd.read_csv(root / "processed" / f"board_{mode}_{asset}.csv.gz", parse_dates=["date", "expiry"],
                        low_memory=False)
    fut = pd.read_csv(root / "raw" / f"futures_{mode}_{asset}.csv.gz")
    futures_secid = fut.drop_duplicates("SHORTNAME").set_index("SHORTNAME")["SECID"].to_dict()

    dates = sorted(board["date"].unique())
    prev = dict(zip(dates[1:], dates[:-1]))
    todo = [d for d in dates if pd.Timestamp(start) <= d <= pd.Timestamp(end) and d in prev]
    traded = board[(board["vol_otm"] > 0) & (board["T"] * 365 >= 1)]
    by_day = dict(tuple(traded.groupby("date")))
    tick = DEFAULT_TICK.get(asset, 1.0)

    def run(d):
        day = by_day.get(d)
        if day is None:
            return pd.DataFrame()
        return synced_trade_iv(client, day, futures_secid, pd.Timestamp(prev[d]), tick)

    with ThreadPoolExecutor(workers) as pool:
        parts = list(tqdm(pool.map(run, todo), total=len(todo), desc=f"sync {asset}"))
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    out.insert(0, "asset", asset)
    name = f"trades_sync_{asset}" if mode == "daily" else f"trades_sync_{mode}_{asset}"
    out.to_csv(root / "processed" / f"{name}.csv.gz", index=False)
    print(f"{asset}: {len(todo)} days, {len(out)} synced contract-days")
    return out
