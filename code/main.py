"""Единственная точка запуска: данные → синхронизация → пилот → основной эксперимент → рисунки.

    python code/main.py              # полный прогон (Si); из кэша ISS около 25 мин
    python code/main.py --quick      # тестовый прогон пилота на одном месяце, результаты в reports/quick/
    python code/main.py --stages figures            # перерисовать рисунки и TeX-таблицы по сохранённым таблицам
    python code/main.py --asset RTS --stages data sync experiment   # проверка переносимости на RTS

Параметры — в code/config.py. Числа и таблицы для статьи и слайдов пишутся в paper/generated/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Расчёты распараллелены по дням; потоки линейной алгебры внутри каждого процесса только мешают.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
import figures  # noqa: E402
from ivs import experiment, pilot, textables  # noqa: E402
from ivs.data import DEFAULT_TICK, build_boards, client_for, daily_dates  # noqa: E402
from ivs.intraday import sync_asset  # noqa: E402

STAGES = ["data", "sync", "pilot", "experiment", "figures"]
PARAMS = ["T_MIN_DAYS", "T_MAX_DAYS", "X_MAX", "GROSS_ERROR", "MIN_POINTS_PER_DAY", "MASKS", "SEEDS",
          "WINGS_Z", "RIDGE", "NOISE", "TAU_LEVEL", "TAU_SLOPE", "TAU_SERIES", "N_BOOT"]
EXP_PARAMS = ["TRAIN_RANGE", "TEST_RANGE", "FAR_KEEP", "TOPN", "SSVI_RIDGE", "SSVI_FSCALE", "LAGS", "LAG_MASKS",
              "LAG_SEEDS", "ARB_Z_MAX", "ARB_STEP", "ARB_TICKS", "ARB_MASK", "ARB_IV_MIN", "ARB_IV_MAX",
              "BREAKDOWN_MASK", "COVER_LEVELS", "EXAMPLE_DATE", "EXAMPLE_MASK", "GAP_MASK"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="короткий тестовый прогон пилота (config.QUICK_RANGE)")
    ap.add_argument("--asset", default=cfg.ASSET, help="базовый актив (по умолчанию config.ASSET)")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    args = ap.parse_args()

    asset = args.asset
    main_asset = asset == cfg.ASSET
    if args.quick:
        mode, tag, (start, end) = "quick", f"{asset}_quick", cfg.QUICK_RANGE
        daily = ((pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), end)
        sync_range = (start, end)
        tables, figs = cfg.ROOT / "reports" / "quick" / "tables", cfg.ROOT / "reports" / "quick" / "figures"
        tex = cfg.ROOT / "reports" / "quick" / "pilot_numbers.tex"
    else:
        mode, tag, (start, end), daily = "daily", asset, cfg.PILOT_RANGE, cfg.DAILY_RANGE
        sync_range = (cfg.TRAIN_RANGE[0], cfg.TEST_RANGE[1])
        tables, figs = cfg.TABLES_DIR, cfg.FIGURES_DIR
        tex = cfg.TEX_NUMBERS if main_asset else tables / f"pilot_numbers_{asset}.tex"
    sync_name = f"trades_sync_{asset}" if mode == "daily" else f"trades_sync_{mode}_{asset}"
    params = {k: getattr(cfg, k) for k in PARAMS}
    t0 = time.time()

    if "data" in args.stages:
        dates = daily_dates(client_for(cfg.DATA_DIR), *daily, cfg.WORKERS_IO)
        build_boards(cfg.DATA_DIR, mode, asset, dates, cfg.WORKERS_IO)
    if "sync" in args.stages:
        sync_asset(cfg.DATA_DIR, asset, *sync_range, cfg.WORKERS_IO, mode=mode)
    if "pilot" in args.stages and (main_asset or args.quick):
        board = pd.read_csv(cfg.DATA_DIR / "processed" / f"board_{mode}_{asset}.csv.gz",
                            parse_dates=["date", "expiry"], low_memory=False)
        sync = pd.read_csv(cfg.DATA_DIR / "processed" / f"{sync_name}.csv.gz", parse_dates=["date", "expiry"])
        obs, res = pilot.run(board, sync, start, end, params, cfg.WORKERS_CPU)
        info = pilot.save(obs, res, tag, start, end, params, cfg.DATA_DIR, tables, tex)
        print(json.dumps({k: info[k] for k in ["period", "days", "points", "gross_removed"]}, ensure_ascii=False))
        print(pd.read_csv(tables / f"pilot_curve_{tag}.csv").query("method in ['best_cb', 'hist', 'hist_cb']")
              .round(3).to_string(index=False))
    if "experiment" in args.stages and not args.quick:
        p = {**params, **{k: getattr(cfg, k) for k in EXP_PARAMS}, "TICK": DEFAULT_TICK[asset]}
        board, sync = experiment.load(cfg.DATA_DIR, asset)
        obs = experiment.observations(board, sync, p)
        models = experiment.train(obs, p, cfg.WORKERS_CPU)
        print(f"train done in {time.time() - t0:.0f} s:", json.dumps({k: round(v, 4) for k, v in models["gp"].items()}))
        context = experiment.day_context(board)
        res = experiment.run(obs, context, p, models, cfg.WORKERS_CPU)
        lags = experiment.run_lags(obs, p, models, cfg.WORKERS_CPU)
        arb = experiment.run_arbitrage(obs, context, p, models, cfg.WORKERS_CPU)
        example = experiment.example_day(obs, context, p, models) if main_asset else None
        info = experiment.save(obs, res, models, lags, arb, asset, p, cfg.DATA_DIR, tables, example)
        print(json.dumps({k: v for k, v in info.items() if k != "params"}, ensure_ascii=False))
        print(pd.read_csv(tables / f"exp_rmse_{asset}.csv").pivot_table(index=["scheme", "mask"], columns="method",
                                                                         values="rmse").round(2).to_string())
    if "figures" in args.stages and (main_asset or args.quick):
        figures.pilot_figures(tables, figs, tag)
        if not args.quick:
            figures.experiment_figures(tables, figs, asset)
            textables.write(tables, cfg.TEX_NUMBERS.parent, asset, cfg.EXTRA_ASSETS)
        print(f"figures -> {figs}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
