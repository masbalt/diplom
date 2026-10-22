"""Единственная точка запуска пилотного эксперимента: данные → синхронизация → пилот → таблицы → рисунки.

    python code/main.py              # полный прогон (Si, 2025-10-01..2026-09-30); из кэша ~25 мин
    python code/main.py --quick      # тестовый прогон на одном месяце, результаты в reports/quick/
    python code/main.py --stages figures   # перерисовать рисунки по сохранённым таблицам

Параметры — в code/config.py. Числа для статьи и слайдов пишутся в paper/generated/pilot_numbers.tex.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
import figures  # noqa: E402
from ivs import pilot  # noqa: E402
from ivs.data import build_boards, client_for, daily_dates  # noqa: E402
from ivs.intraday import sync_asset  # noqa: E402

STAGES = ["data", "sync", "pilot", "figures"]
PARAMS = ["T_MIN_DAYS", "T_MAX_DAYS", "X_MAX", "GROSS_ERROR", "MIN_POINTS_PER_DAY", "MASKS", "SEEDS",
          "WINGS_Z", "RIDGE", "NOISE", "TAU_LEVEL", "TAU_SLOPE", "TAU_SERIES", "N_BOOT"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="короткий тестовый прогон (config.QUICK_RANGE)")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    args = ap.parse_args()

    asset = cfg.ASSET
    if args.quick:
        mode, tag, (start, end) = "quick", f"{asset}_quick", cfg.QUICK_RANGE
        daily = ((pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), end)
        tables, figs = cfg.ROOT / "reports" / "quick" / "tables", cfg.ROOT / "reports" / "quick" / "figures"
        tex = cfg.ROOT / "reports" / "quick" / "pilot_numbers.tex"
    else:
        mode, tag, (start, end), daily = "daily", asset, cfg.PILOT_RANGE, cfg.DAILY_RANGE
        tables, figs, tex = cfg.TABLES_DIR, cfg.FIGURES_DIR, cfg.TEX_NUMBERS
    sync_name = f"trades_sync_{asset}" if mode == "daily" else f"trades_sync_{mode}_{asset}"
    params = {k: getattr(cfg, k) for k in PARAMS}
    t0 = time.time()

    if "data" in args.stages:
        dates = daily_dates(client_for(cfg.DATA_DIR), *daily, cfg.WORKERS_IO)
        build_boards(cfg.DATA_DIR, mode, asset, dates, cfg.WORKERS_IO)
    if "sync" in args.stages:
        sync_asset(cfg.DATA_DIR, asset, start, end, cfg.WORKERS_IO, mode=mode)
    if "pilot" in args.stages:
        board = pd.read_csv(cfg.DATA_DIR / "processed" / f"board_{mode}_{asset}.csv.gz",
                            parse_dates=["date", "expiry"], low_memory=False)
        sync = pd.read_csv(cfg.DATA_DIR / "processed" / f"{sync_name}.csv.gz", parse_dates=["date", "expiry"])
        obs, res = pilot.run(board, sync, start, end, params, cfg.WORKERS_CPU)
        info = pilot.save(obs, res, tag, start, end, params, cfg.DATA_DIR, tables, tex)
        print(json.dumps({k: info[k] for k in ["period", "days", "points", "gross_removed"]}, ensure_ascii=False))
        print(pd.read_csv(tables / f"pilot_curve_{tag}.csv").query("method in ['best_cb', 'hist', 'hist_cb']")
              .round(3).to_string(index=False))
    if "figures" in args.stages:
        figures.pilot_figures(tables, figs, tag)
        print(f"figures -> {figs}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
