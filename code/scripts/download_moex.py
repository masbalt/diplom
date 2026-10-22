"""Загрузка итогов торгов опционами MOEX и сборка дневных досок (CLI над ivs.data).

Пример:
    python code/scripts/download_moex.py --data-dir data \
        --monthly 2012-01:2026-09 --monthly-assets Si RTS BR MIX GOLD CNY \
        --daily 2024-10-01:2026-09-30 --daily-assets Si RTS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivs.data import build_boards, client_for, daily_dates, monthly_dates  # noqa: E402


def parse_range(text: str) -> tuple[str, str]:
    a, b = text.split(":")
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--monthly", type=parse_range)
    ap.add_argument("--monthly-assets", nargs="*", default=[])
    ap.add_argument("--daily", type=parse_range)
    ap.add_argument("--daily-assets", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.data_dir)
    client = client_for(root)
    if args.monthly:
        dates = monthly_dates(client, *args.monthly, args.workers)
        print(f"monthly snapshots: {len(dates)} dates {dates[0]}..{dates[-1]}")
        for asset in args.monthly_assets:
            build_boards(root, "monthly", asset, dates, args.workers)
    if args.daily:
        dates = daily_dates(client, *args.daily, args.workers)
        print(f"daily: {len(dates)} trading days {dates[0]}..{dates[-1]}")
        for asset in args.daily_assets:
            build_boards(root, "daily", asset, dates, args.workers)


if __name__ == "__main__":
    main()
