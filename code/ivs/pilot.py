"""Пилот: помогает ли история поверхности восстанавливать скрытые сделки (только вычисления).

Наблюдения дня D — реальные сделки по OTM-опционам с IV, синхронизированной с фьючерсом
по минутам. Часть наблюдений скрывается, методы из ivs.recon восстанавливают их по
оставшимся точкам (и, для history-aware, по вчерашней расчётной кривой MOEX).

Схемы скрытия:
  random — доля m из MASKS случайных точек (SEEDS повторов на день, seed = YYYYMMDD);
  wings  — скрыты все точки с |z| > WINGS_Z, видны только точки около ATM;
  slice  — скрыта целиком одна серия (по очереди каждая), видны остальные.

``moex_today`` — расчётная кривая дня D, построенная по полному стакану дня D:
справочный уровень, а не конкурент в разреженной постановке.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from . import recon

METHODS = ["hist", "cb_quad", "cb_svi", "hist_cb", "moex_today"]
CURRENT_BOARD = ["cb_quad", "cb_svi"]


# --- наблюдения ---------------------------------------------------------------

def settle_smiles(board: pd.DataFrame) -> dict:
    ok = board["iv_settle"].notna() & (board["iv_settle_err"] < 0.01) & (board["T"] * 365 >= 1)
    return {d: recon.Smiles(g) for d, g in board[ok].groupby("date")}


def build_observations(board: pd.DataFrame, sync: pd.DataFrame, p: dict) -> pd.DataFrame:
    obs = sync.merge(board[["date", "expiry", "strike", "T", "z"]], on=["date", "expiry", "strike"])
    obs = obs.rename(columns={"iv_sync": "iv", "k_sync": "k"})
    obs["x"] = obs["k"] / np.sqrt(obs["T"])
    obs["sqrtT"] = np.sqrt(obs["T"])
    days = obs["T"] * 365
    obs = obs[(days >= p["T_MIN_DAYS"]) & (days <= p["T_MAX_DAYS"]) & (obs["x"].abs() <= p["X_MAX"])]

    smiles = settle_smiles(board)
    dates = sorted(board["date"].unique())
    prev = dict(zip(dates[1:], dates[:-1]))
    hist, today = np.full(len(obs), np.nan), np.full(len(obs), np.nan)
    for (d, e), g in obs.groupby(["date", "expiry"]):
        idx = obs.index.get_indexer(g.index)
        T = float(g["T"].iloc[0])
        if d in prev and prev[d] in smiles:
            hist[idx] = smiles[prev[d]](e, g["k"].to_numpy(), T)
        if d in smiles:
            today[idx] = smiles[d](e, g["k"].to_numpy(), T)
    obs["hist"], obs["today"] = hist, today
    obs = obs.dropna(subset=["hist", "today", "iv"])
    # Очистка грубых ошибок сделок (≈0.5 % точек): одинакова для всех методов.
    gross = (obs["iv"] - obs["today"]).abs() > p["GROSS_ERROR"]
    obs = obs[~gross]
    n = obs.groupby("date")["iv"].transform("size")
    out = obs[n >= p["MIN_POINTS_PER_DAY"]].reset_index(drop=True)
    out.attrs["gross_removed"] = int(gross.sum())
    return out


# --- эксперимент --------------------------------------------------------------

def predict(vis: pd.DataFrame, hid: pd.DataFrame, p: dict) -> dict[str, np.ndarray]:
    prior = (p["NOISE"], p["TAU_LEVEL"], p["TAU_SLOPE"], p["TAU_SERIES"])
    return {
        "hist": hid["hist"].to_numpy(),
        "cb_quad": recon.cb_quad(vis, hid, p["RIDGE"]),
        "cb_svi": recon.cb_svi(vis, hid, p["RIDGE"]),
        "hist_cb": recon.hist_cb(vis, hid, prior),
        "moex_today": hid["today"].to_numpy(),
    }


def run_day(obs: pd.DataFrame, p: dict) -> list[pd.DataFrame]:
    obs = obs.reset_index(drop=True)
    n = len(obs)
    rng = np.random.default_rng(int(obs["date"].iloc[0].strftime("%Y%m%d")))
    configs = []
    for m in p["MASKS"]:
        for seed in range(p["SEEDS"]):
            n_hid = int(np.clip(round(m * n), 1, n - 1))
            configs.append(("random", m, seed, rng.choice(n, n_hid, replace=False)))
    wings = np.flatnonzero(obs["z"].abs() > p["WINGS_Z"])
    if 0 < len(wings) < n:
        configs.append(("wings", np.nan, 0, wings))
    for i, e in enumerate(obs["expiry"].unique()):
        idx = np.flatnonzero(obs["expiry"] == e)
        if 2 <= len(idx) < n:
            configs.append(("slice", np.nan, i, idx))

    out = []
    for scheme, m, seed, hid_idx in configs:
        mask = np.zeros(n, bool)
        mask[hid_idx] = True
        vis, hid = obs[~mask], obs[mask].reset_index(drop=True)
        res = hid[["date", "expiry", "strike", "T", "z", "iv"]].copy()
        res["scheme"], res["mask"], res["seed"] = scheme, m, seed
        res["n_obs"], res["n_visible"] = n, len(vis)
        res["n_visible_series"] = hid["expiry"].map(vis["expiry"].value_counts()).fillna(0).to_numpy()
        for name, pred in predict(vis, hid, p).items():
            res[name] = pred
        out.append(res)
    return out


def run(board: pd.DataFrame, sync: pd.DataFrame, start: str, end: str, p: dict, workers: int):
    """Возвращает (наблюдения, результаты по скрытым точкам)."""
    sync = sync[(sync["date"] >= pd.Timestamp(start)) & (sync["date"] <= pd.Timestamp(end))]
    obs = build_observations(board, sync, p)
    days = [g for _, g in obs.groupby("date")]
    with ProcessPoolExecutor(workers) as pool:
        parts = pool.map(partial(run_day, p=p), days, chunksize=4)
        res = pd.concat([df for day in parts for df in day], ignore_index=True)
    return obs, res


# --- сводка -------------------------------------------------------------------

def _select(res: pd.DataFrame, scheme: str, mask: float | None) -> pd.DataFrame:
    r = res[res["scheme"] == scheme]
    return r if mask is None else r[np.isclose(r["mask"], mask)]


def _day_sums(r: pd.DataFrame, methods: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sq = pd.DataFrame({m: (100 * (r[m] - r["iv"])) ** 2 for m in methods})
    sq["date"] = r["date"].to_numpy()
    g = sq.groupby("date")
    return g[methods].sum().to_numpy(), g.size().to_numpy()


def rmse_with_ci(r: pd.DataFrame, methods: list[str], n_boot: int, seed: int = 0) -> pd.DataFrame:
    """RMSE (п.в.) каждого метода и 95 % интервал бутстрепом по дням (дни ресэмплируются парно)."""
    S, C = _day_sums(r, methods)
    idx = np.random.default_rng(seed).integers(0, len(S), (n_boot, len(S)))
    boot = np.sqrt(S[idx].sum(axis=1) / C[idx].sum(axis=1)[:, None])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"method": methods, "rmse": np.sqrt(S.sum(axis=0) / C.sum()), "lo": lo, "hi": hi})


def diff_with_ci(r: pd.DataFrame, a: str, b: str, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """RMSE(a) − RMSE(b), п.в., с бутстрепом по дням."""
    S, C = _day_sums(r, [a, b])
    idx = np.random.default_rng(seed).integers(0, len(S), (n_boot, len(S)))
    tot = S[idx].sum(axis=1) / C[idx].sum(axis=1)[:, None]
    boot = np.sqrt(tot[:, 0]) - np.sqrt(tot[:, 1])
    point = np.sqrt(S[:, 0].sum() / C.sum()) - np.sqrt(S[:, 1].sum() / C.sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    """RMSE и MAE всех методов по каждой схеме и доле скрытия."""
    rows = []
    for (scheme, m), r in res.groupby(["scheme", res["mask"].fillna(-1)]):
        row = {"scheme": scheme, "mask": np.nan if m < 0 else m, "n_points": len(r), "days": r["date"].nunique()}
        for meth in METHODS:
            err = 100 * (r[meth] - r["iv"])
            row[f"rmse_{meth}"] = float(np.sqrt(np.mean(err**2)))
            row[f"mae_{meth}"] = float(err.abs().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def curve_table(res: pd.DataFrame, masks: list[float], n_boot: int) -> pd.DataFrame:
    """Данные главного рисунка: RMSE с интервалами по доле скрытия (схема random).

    ``best_cb`` — лучший по RMSE метод только текущей доски при данной доле скрытия.
    """
    rows = []
    for m in masks:
        r = _select(res, "random", m)
        t = rmse_with_ci(r, METHODS, n_boot).set_index("method")
        best = t.loc[CURRENT_BOARD, "rmse"].idxmin()
        visible = float(r.drop_duplicates(["date", "seed"])["n_visible"].median())
        for meth in METHODS + ["best_cb"]:
            src = best if meth == "best_cb" else meth
            rows.append({"mask": m, "method": meth, "source": src, "visible_median": visible,
                         **t.loc[src, ["rmse", "lo", "hi"]].to_dict()})
    return pd.DataFrame(rows)


def by_visible(res: pd.DataFrame) -> pd.DataFrame:
    """RMSE в зависимости от числа видимых сделок в той же серии (схема random)."""
    rr = res[res["scheme"] == "random"].copy()
    rr["vis_bin"] = pd.cut(rr["n_visible_series"], [-0.5, 0.5, 2.5, 4.5, 9.5, 1e9],
                           labels=["0", "1-2", "3-4", "5-9", "10+"])
    return rr.groupby("vis_bin", observed=True).apply(
        lambda g: pd.Series({"n": len(g), **{m: float(np.sqrt(np.mean((100 * (g[m] - g["iv"])) ** 2)))
                                             for m in METHODS}}),
        include_groups=False,
    )


def comparisons(res: pd.DataFrame, masks: list[float], n_boot: int) -> dict:
    """Разности RMSE history-aware vs лучший метод текущей доски и vs чистая история."""
    out = {}
    for scheme, m in [("random", x) for x in masks] + [("wings", None), ("slice", None)]:
        r = _select(res, scheme, m)
        best = min(CURRENT_BOARD, key=lambda meth: np.sqrt(np.mean((r[meth] - r["iv"]) ** 2)))
        out[f"{scheme}_{m}"] = {
            "best_cb": best,
            "hist_cb_minus_best_cb": diff_with_ci(r, "hist_cb", best, n_boot),
            "hist_cb_minus_hist": diff_with_ci(r, "hist_cb", "hist", n_boot),
        }
    return out


def save(obs: pd.DataFrame, res: pd.DataFrame, asset: str, start: str, end: str, p: dict,
         data_dir: Path, tables_dir: Path, tex_path: Path) -> dict:
    """Сохраняет результаты, таблицы и TeX-макросы с числами для статьи и слайда."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(data_dir) / "processed" / f"pilot_results_{asset}.csv.gz", index=False)
    summarize(res).to_csv(tables_dir / f"pilot_summary_{asset}.csv", index=False)
    curve = curve_table(res, p["MASKS"], p["N_BOOT"])
    curve.to_csv(tables_dir / f"pilot_curve_{asset}.csv", index=False)
    by_visible(res).to_csv(tables_dir / f"pilot_by_visible_{asset}.csv")
    info = {
        "asset": asset,
        "period": f"{obs['date'].min().date()}..{obs['date'].max().date()}",
        "requested_period": f"{start}..{end}",
        "days": int(obs["date"].nunique()),
        "points": len(obs),
        "gross_removed": obs.attrs.get("gross_removed"),
        "points_per_day_median": float(obs.groupby("date").size().median()),
        "series_per_day_median": float(obs.groupby("date")["expiry"].nunique().median()),
        "intraday_iv_std_vp_median": float(100 * obs["iv_sync_std"].median()),
        "comparisons_rmse_diff_vp": comparisons(res, p["MASKS"], p["N_BOOT"]),
        "params": p,
    }
    (tables_dir / f"pilot_info_{asset}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    write_tex(curve, info, tex_path)
    return info


def _num(x: float, digits: int = 2) -> str:
    """Число для русского TeX: десятичная запятая, тонкий пробел между тысячами."""
    s = f"{x:,.{digits}f}".replace(",", " ").replace(".", "{,}")
    return s.replace(" ", "\\,")


def write_tex(curve: pd.DataFrame, info: dict, path: Path) -> None:
    lo_m, hi_m = min(curve["mask"]), max(curve["mask"])

    def val(m, meth, col="rmse"):
        return float(curve[(curve["mask"] == m) & (curve["method"] == meth)][col].iloc[0])

    cmp_lo = info["comparisons_rmse_diff_vp"][f"random_{lo_m}"]["hist_cb_minus_best_cb"]
    cmp_hi = info["comparisons_rmse_diff_vp"][f"random_{hi_m}"]["hist_cb_minus_best_cb"]
    macros = {
        "PilotAsset": info["asset"],
        "PilotPeriod": info["period"].replace("..", "--"),
        "PilotDays": str(info["days"]),
        "PilotTrades": _num(info["points"], 0),
        "PilotTradesPerDay": _num(info["points_per_day_median"], 0),
        "PilotMaskLow": _num(100 * lo_m, 0),
        "PilotMaskHigh": _num(100 * hi_m, 0),
        "PilotCbLow": _num(val(lo_m, "best_cb")),
        "PilotHistCbLow": _num(val(lo_m, "hist_cb")),
        "PilotCbHigh": _num(val(hi_m, "best_cb")),
        "PilotHistCbHigh": _num(val(hi_m, "hist_cb")),
        "PilotHistHigh": _num(val(hi_m, "hist")),
        "PilotDiffLow": _num(cmp_lo[0]),
        "PilotDiffLowCI": f"[{_num(cmp_lo[1])};\\ {_num(cmp_lo[2])}]",
        "PilotDiffHigh": _num(cmp_hi[0]),
        "PilotDiffHighCI": f"[{_num(cmp_hi[1])};\\ {_num(cmp_hi[2])}]",
        "PilotVisibleLow": _num(val(lo_m, "best_cb", "visible_median"), 0),
        "PilotVisibleHigh": _num(val(hi_m, "best_cb", "visible_median"), 0),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Сгенерировано code/main.py (этап tables) — не редактировать вручную."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    path.write_text("\n".join(lines) + "\n")
