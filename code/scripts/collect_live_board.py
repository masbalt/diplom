"""Снимок текущей опционной доски MOEX с лучшими bid/ask.

Исторические bid/ask и стаканы по опционам в бесплатном ISS недоступны (только по подписке),
но текущие лучшие котировки отдаются с задержкой ~15 мин. Регулярный запуск этого скрипта
(например, cron раз в час в торговые часы) накапливает собственную историю котировок.

Пример:
    python code/scripts/collect_live_board.py --data-dir data --assets Si RTS

Пишет data/live/{asset}/{YYYY-MM-DD_HHMMSS}.csv.gz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivs.iss import ISSClient  # noqa: E402

SEC_COLS = ["SECID", "ASSETCODE", "SHORTNAME", "STRIKE", "OPTIONTYPE", "LASTTRADEDATE", "UNDERLYINGASSET",
            "UNDERLYINGSETTLEPRICE", "PREVSETTLEPRICE", "MINSTEP"]
MD_COLS = ["SECID", "BID", "OFFER", "LAST", "NUMTRADES", "VOLTODAY", "OPENPOSITION", "UPDATETIME", "SYSTIME",
           "TRADE_SESSION_DATE"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--assets", nargs="+", default=["Si", "RTS"])
    args = ap.parse_args()

    client = ISSClient()
    data = client.get_json("engines/futures/markets/options/securities", cache=False)
    sec = pd.DataFrame(data["securities"]["data"], columns=data["securities"]["columns"])[SEC_COLS]
    md = pd.DataFrame(data["marketdata"]["data"], columns=data["marketdata"]["columns"])[MD_COLS]
    board = sec.merge(md, on="SECID")

    fut = client.get_json("engines/futures/markets/forts/securities", cache=False)
    fmd = pd.DataFrame(fut["marketdata"]["data"], columns=fut["marketdata"]["columns"])
    fmd = fmd[["SECID", "BID", "OFFER", "LAST", "SYSTIME"]].add_prefix("UND_")
    board = board.merge(fmd, left_on="UNDERLYINGASSET", right_on="UND_SECID", how="left")

    stamp = pd.Timestamp.now(tz="Europe/Moscow").strftime("%Y-%m-%d_%H%M%S")
    for asset in args.assets:
        part = board[board["ASSETCODE"] == asset]
        out = Path(args.data_dir) / "live" / asset
        out.mkdir(parents=True, exist_ok=True)
        part.to_csv(out / f"{stamp}.csv.gz", index=False)
        quoted = ((part["BID"] > 0) & (part["OFFER"] > 0)).sum()
        print(f"{asset}: {len(part)} options, {quoted} with two-sided quotes -> {out / stamp}.csv.gz")


if __name__ == "__main__":
    main()
