"""IV сделок, синхронизированная по минутным свечам (CLI над ivs.intraday.sync_asset).

Пример:
    python code/scripts/sync_trades.py --data-dir data --asset Si --from 2025-10-01 --to 2026-09-30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivs.intraday import sync_asset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--asset", default="Si")
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    sync_asset(Path(args.data_dir), args.asset, args.start, args.end, args.workers)


if __name__ == "__main__":
    main()
