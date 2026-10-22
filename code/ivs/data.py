"""Загрузка итогов торгов MOEX ISS и сборка дневных досок (используется main.py и scripts/download_moex.py).

Файлы в ``root``:
    raw/options_{mode}_{asset}.csv.gz    — сырые итоги дня ISS по опционам;
    raw/futures_{mode}_{asset}.csv.gz    — итоги дня по фьючерсам (расчётная цена, SHORTNAME);
    meta/series.csv                      — справочник серий (экспирация, базовый фьючерс);
    processed/board_{mode}_{asset}.csv.gz — доски в OTM-соглашении (см. ivs.board).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .board import build_board
from .contracts import SeriesRegistry, parse_secids
from .iss import ISSClient

# Шаг цены опционов (пункты) по текущей спецификации — только для оценки
# неопределённости IV из-за округления цены.
DEFAULT_TICK = {"Si": 1.0, "RTS": 10.0, "BR": 0.01, "MIX": 25.0, "GOLD": 0.1, "CNY": 0.001}
REFERENCE_ASSET = "Si"  # по нему определяются торговые дни


def client_for(root: Path) -> ISSClient:
    return ISSClient(Path(root) / "cache" / "iss")


def trading_days(client: ISSClient, days: list[pd.Timestamp], workers: int) -> list[str]:
    def has_data(d: pd.Timestamp) -> bool:
        data = client.get_json(
            "history/engines/futures/markets/options/securities",
            {"date": d.strftime("%Y-%m-%d"), "assetcode": REFERENCE_ASSET, "start": 0},
        )
        return len(data["history"]["data"]) > 0

    with ThreadPoolExecutor(workers) as pool:
        flags = list(tqdm(pool.map(has_data, days), total=len(days), desc="calendar"))
    return [d.strftime("%Y-%m-%d") for d, f in zip(days, flags) if f]


def daily_dates(client: ISSClient, start: str, end: str, workers: int) -> list[str]:
    return trading_days(client, list(pd.date_range(start, end, freq="D")), workers)


def monthly_dates(client: ISSClient, start: str, end: str, workers: int) -> list[str]:
    """Первый торговый день не раньше 15-го числа каждого месяца."""
    months = pd.date_range(f"{start}-01", f"{end}-01", freq="MS")
    candidates = [m + pd.Timedelta(days=offset) for m in months for offset in range(14, 24)]
    found = set(trading_days(client, candidates, workers))
    out = []
    for m in months:
        for offset in range(14, 24):
            d = (m + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            if d in found:
                out.append(d)
                break
    return out


def download(client: ISSClient, kind: str, asset: str, dates: list[str], workers: int) -> pd.DataFrame:
    fetch = client.options_history if kind == "options" else client.futures_history
    with ThreadPoolExecutor(workers) as pool:
        frames = list(tqdm(pool.map(lambda d: fetch(d, asset), dates), total=len(dates), desc=f"{kind} {asset}"))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_futures(board: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
    """Расчётная цена базового фьючерса (по SHORTNAME вида ``Si-12.26``) для сверки F."""
    if fut.empty or board.empty:
        board["F_fut"] = float("nan")
        return board
    f = fut[["TRADEDATE", "SHORTNAME", "SETTLEPRICE"]].copy()
    f["date"] = pd.to_datetime(f["TRADEDATE"])
    f = f.rename(columns={"SHORTNAME": "underlying", "SETTLEPRICE": "F_fut"}).drop(columns="TRADEDATE")
    f = f.drop_duplicates(["date", "underlying"])
    return board.merge(f, on=["date", "underlying"], how="left")


def build_boards(root: Path, mode: str, asset: str, dates: list[str], workers: int) -> pd.DataFrame:
    """Скачивает (или берёт из кэша) итоги дня и сохраняет доски ``processed/board_{mode}_{asset}``."""
    root = Path(root)
    for sub in ("raw", "meta", "processed"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    client = client_for(root)
    registry = SeriesRegistry(root / "meta" / "series.csv", client)

    raw = download(client, "options", asset, dates, workers)
    fut = download(client, "futures", asset, dates, workers)
    raw.to_csv(root / "raw" / f"options_{mode}_{asset}.csv.gz", index=False)
    fut.to_csv(root / "raw" / f"futures_{mode}_{asset}.csv.gz", index=False)
    if raw.empty:
        print(f"{mode} {asset}: no option data")
        return pd.DataFrame()

    parsed = parse_secids(raw["SECID"])
    examples = raw.assign(group=parsed["group"]).dropna(subset=["group"]).groupby("group")["SECID"].first()
    registry.resolve(examples.to_dict(), workers=workers)
    lookup = registry.lookup()

    tick = DEFAULT_TICK.get(asset, 1.0)
    boards = [build_board(g, lookup, tick) for _, g in tqdm(raw.groupby("TRADEDATE"), desc=f"boards {asset}")]
    board = pd.concat([b for b in boards if not b.empty], ignore_index=True)
    board.insert(0, "asset", asset)
    board = attach_futures(board, fut)
    board.to_csv(root / "processed" / f"board_{mode}_{asset}.csv.gz", index=False)
    print(f"{mode} {asset}: raw={len(raw)} unparsed_secids={int(parsed['group'].isna().sum())} board_rows={len(board)}")
    return board
