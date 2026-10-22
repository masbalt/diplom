"""Аудит данных MOEX по опционам: плотность доски, разреженность, качество наблюдений, арбитраж.

Пример:
    python code/scripts/audit_moex.py --data-dir data --out reports

Читает доски data/processed/board_{monthly,daily}_*.csv.gz и trades_sync_*.csv.gz,
пишет таблицы в reports/tables/ и графики в reports/figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivs import black76, moex_curve  # noqa: E402
from figures import INK, INK_2, SERIES, style  # noqa: E402
from ivs.data import DEFAULT_TICK  # noqa: E402

# Рабочая зона поверхности: стандартизованная денежность и срок.
Z_MAX = 2.0
T_MIN_DAYS, T_MAX_DAYS = 7, 365


def load(root: Path, mode: str, asset: str) -> pd.DataFrame:
    path = root / "processed" / f"board_{mode}_{asset}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date", "expiry"], low_memory=False)


def in_zone(b: pd.DataFrame) -> pd.Series:
    days = b["T"] * 365
    return (b["z"].abs() <= Z_MAX) & (days >= T_MIN_DAYS) & (days <= T_MAX_DAYS)


def daily_density(b: pd.DataFrame) -> pd.DataFrame:
    """Показатели плотности доски по дням."""
    b = b.assign(
        zone=in_zone(b),
        listed=(b["settle_c"] > 0) | (b["settle_p"] > 0),
        has_oi=(b["oi_c"] + b["oi_p"]) > 0,
        traded_any=(b["vol_c"] + b["vol_p"]) > 0,
        traded_otm=(b["vol_otm"] > 0) & b["iv_trade"].notna(),
    )
    z = b[b["zone"]]
    per_series = z.groupby(["date", "expiry"])["traded_otm"].sum()
    out = pd.DataFrame({
        "series": b[b["T"] * 365 >= 1].groupby("date")["expiry"].nunique(),
        "series_zone": z.groupby("date")["expiry"].nunique(),
        "listed": b.groupby("date")["listed"].sum(),
        "listed_zone": z.groupby("date")["listed"].sum(),
        "oi_zone": z.groupby("date")["has_oi"].sum(),
        "traded_any": b.groupby("date")["traded_any"].sum(),
        "traded_otm_zone": z.groupby("date")["traded_otm"].sum(),
        "series_ge5": (per_series >= 5).groupby("date").sum(),
        "series_ge1": (per_series >= 1).groupby("date").sum(),
        "volume": b.groupby("date").apply(lambda g: (g["vol_c"] + g["vol_p"]).sum(), include_groups=False),
        "atm30": b.groupby("date").apply(atm_constant_maturity, include_groups=False),
    }).fillna(0)
    out["coverage_zone"] = out["traded_otm_zone"] / out["listed_zone"].replace(0, np.nan)
    return out


def atm_constant_maturity(g: pd.DataFrame, days: float = 30.0) -> float:
    """ATM IV на постоянном сроке: линейная интерполяция полной дисперсии по T."""
    s = g.groupby("expiry")[["T", "atm_iv"]].first().dropna()
    s = s[s["T"] * 365 >= 3].sort_values("T")
    if len(s) < 2:
        return np.nan
    w = s["atm_iv"] ** 2 * s["T"]
    t = days / 365
    if s["T"].iloc[0] <= t <= s["T"].iloc[-1]:
        return float(np.sqrt(np.interp(t, s["T"], w) / t))
    # Без брекета (старые годы без недельных серий) — ATM ближайшей серии не дальше 60 дней.
    if t < s["T"].iloc[0] <= 60 / 365:
        return float(s["atm_iv"].iloc[0])
    return np.nan


def persistence(b: pd.DataFrame) -> pd.DataFrame:
    """Доля сегодняшних наблюдаемых точек (серия, страйк), которые торговались и вчера."""
    obs = b[in_zone(b) & (b["vol_otm"] > 0) & b["iv_trade"].notna()]
    sets = obs.groupby("date").apply(lambda g: set(zip(g["expiry"], g["strike"])), include_groups=False)
    rows = []
    for (d0, s0), (d1, s1) in zip(sets.items(), list(sets.items())[1:]):
        if s1:
            rows.append({"date": d1, "n": len(s1), "seen_yesterday": len(s1 & s0) / len(s1),
                         "jaccard": len(s1 & s0) / len(s1 | s0)})
    return pd.DataFrame(rows)


def settle_vs_curve(b: pd.DataFrame, n_days: int = 40) -> pd.DataFrame:
    """Насколько точно IV расчётных цен описывается 6-параметрической кривой НКЦ."""
    rows = []
    dates = sorted(b["date"].unique())
    for d in dates[:: max(1, len(dates) // n_days)]:
        day = b[(b["date"] == d) & (b["T"] * 365 >= T_MIN_DAYS)]
        for e, g in day.groupby("expiry"):
            g = g[g["iv_settle"].notna() & (g["iv_settle_err"] < 0.002)]
            if len(g) < 8:
                continue
            x = g["k"] / np.sqrt(g["T"])
            _, rmse = moex_curve.fit(x.to_numpy(), g["T"].iloc[0], 100 * g["iv_settle"].to_numpy())
            rows.append({"date": d, "expiry": e, "n": len(g), "rmse_vp": rmse})
    return pd.DataFrame(rows)


def arbitrage(b: pd.DataFrame, tick: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Статический арбитраж в расчётных ценах MOEX.

    Бабочка: выпуклость цены колла по страйку с допуском на округление до шага цены.
    Календарь: полная дисперсия w(k) = sigma^2 T не убывает по T при фиксированном k.
    """
    fly, cal = [], []
    for (d, e), g in b[b["T"] * 365 >= 1].groupby(["date", "expiry"]):
        g = g.sort_values("strike")
        g = g[(g["settle_c"] > 0)]
        if len(g) < 3:
            continue
        K, C = g["strike"].to_numpy(), g["settle_c"].to_numpy()
        slope = np.diff(C) / np.diff(K)
        tol = 2 * tick / np.diff(K)
        viol = np.diff(slope) < -(tol[1:] + tol[:-1]) / 2
        fly.append({"date": d, "expiry": e, "n_triplets": len(viol), "n_viol": int(viol.sum())})
    for d, day in b[(b["T"] * 365 >= T_MIN_DAYS) & b["iv_settle"].notna()].groupby("date"):
        series = [g.sort_values("k") for _, g in day.groupby("expiry")]
        series.sort(key=lambda g: g["T"].iloc[0])
        for g1, g2 in zip(series, series[1:]):
            # Сравнение в рабочей зоне ближней серии |z| <= Z_MAX.
            half = Z_MAX * g1["atm_iv"].iloc[0] * np.sqrt(g1["T"].iloc[0])
            lo = max(g1["k"].min(), g2["k"].min(), -half)
            hi = min(g1["k"].max(), g2["k"].max(), half)
            if not hi > lo:
                continue
            grid = np.linspace(lo, hi, 41)
            T1, T2 = g1["T"].iloc[0], g2["T"].iloc[0]
            w1 = np.interp(grid, g1["k"], g1["iv_settle"] ** 2 * T1)
            iv2 = np.interp(grid, g2["k"], g2["iv_settle"])
            # На сколько п.в. нужно поднять IV дальней серии, чтобы w(T2) >= w(T1).
            bump = 100 * (np.sqrt(w1 / T2) - iv2)
            cal.append({"date": d, "T1": T1, "T2": T2, "share_viol": float((bump > 0.01).mean()),
                        "max_bump_vp": float(max(bump.max(), 0.0))})
    return pd.DataFrame(fly), pd.DataFrame(cal)


GRID_X = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
GRID_T_DAYS = np.array([14, 30, 60, 90])


def fixed_grid(b: pd.DataFrame) -> pd.DataFrame:
    """Расчётная поверхность на фиксированной сетке x = k/sqrt(T) × срок (интерполяция w по T)."""
    rows = {}
    ok = b["iv_settle"].notna() & (b["iv_settle_err"] < 0.01) & (b["T"] * 365 >= 3)
    for d, day in b[ok].groupby("date"):
        series = []
        for _, g in day.groupby("expiry"):
            g = g.sort_values("k")
            T = g["T"].iloc[0]
            x = g["k"] / np.sqrt(T)
            if x.min() > GRID_X[0] or x.max() < GRID_X[-1]:
                continue
            series.append((T, np.interp(GRID_X, x, g["iv_settle"]) ** 2 * T))
        if len(series) < 2:
            continue
        series.sort(key=lambda s: s[0])
        Ts = np.array([s[0] for s in series])
        W = np.array([s[1] for s in series])
        t = GRID_T_DAYS / 365
        if Ts[0] > t[0] or Ts[-1] < t[-1]:
            continue
        sig = np.array([[np.sqrt(np.interp(tt, Ts, W[:, j]) / tt) for j in range(len(GRID_X))] for tt in t])
        rows[d] = sig.ravel()
    cols = [f"T{tt}_x{x:+.1f}" for tt in GRID_T_DAYS for x in GRID_X]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols).sort_index()


def surface_dynamics(grid: pd.DataFrame) -> dict:
    """Дневные изменения поверхности: масштаб и доля общих факторов (PCA)."""
    dates = grid.index.to_series()
    consecutive = dates.diff().dt.days <= 4
    dg = grid.diff()[consecutive].dropna()
    centered = dg - dg.mean()
    _, s, _ = np.linalg.svd(centered.to_numpy(), full_matrices=False)
    share = s**2 / np.sum(s**2)
    atm = [c for c in grid.columns if c.endswith("x+0.0")]
    return {
        "days": len(dg),
        "rms_daily_change_vp": float(100 * np.sqrt(np.mean(dg.to_numpy() ** 2))),
        "rms_daily_change_atm_vp": float(100 * np.sqrt(np.mean(dg[atm].to_numpy() ** 2))),
        "median_abs_daily_change_vp": float(100 * np.median(np.abs(dg.to_numpy()))),
        "median_abs_daily_change_atm30_vp": float(100 * dg["T30_x+0.0"].abs().median()),
        "pca_explained_top3": [float(v) for v in share[:3]],
        "lag1_autocorr_atm30": float(dg["T30_x+0.0"].autocorr()),
    }


def sync_quality(b: pd.DataFrame, sync: pd.DataFrame) -> pd.DataFrame:
    """Отклонение IV сделок от IV расчётной цены дня: наивная дневная VWAP vs синхронизация."""
    m = sync.merge(b, on=["date", "expiry", "strike"], how="inner")
    m = m[in_zone(m)]
    return pd.DataFrame({
        "date": m["date"], "z": m["z"], "T_days": m["T"] * 365, "vol": m["vol_otm"],
        "naive_vp": 100 * (m["iv_trade"] - m["iv_settle"]),
        "sync_vp": 100 * (m["iv_sync"] - m["iv_settle"]),
        "intraday_std_vp": 100 * m["iv_sync_std"],
    })


# --- графики ------------------------------------------------------------------

def fig_density(monthly: dict[str, pd.DataFrame], out: Path) -> None:
    assets = [a for a, d in monthly.items() if not d.empty]
    ncol = 3
    nrow = int(np.ceil(len(assets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.6 * nrow), sharex=True)
    for ax, a in zip(axes.flat, assets):
        d = monthly[a]
        ax.plot(d.index, d["listed_zone"], color=SERIES[0], label="в листинге")
        ax.plot(d.index, d["oi_zone"], color=SERIES[1], label="с открытым интересом")
        ax.plot(d.index, d["traded_otm_zone"], color=SERIES[2], label="со сделками (OTM)")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(a)
    for ax in list(axes.flat)[len(assets):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Точек поверхности в рабочей зоне (|z| ≤ {Z_MAX:g}, {T_MIN_DAYS}–{T_MAX_DAYS} дн.), "
                 "срез в середине месяца", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "density_monthly.png")
    plt.close(fig)


def fig_daily_points(daily: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for i, (a, d) in enumerate(daily.items()):
        axes[0].plot(d.index, d["traded_otm_zone"], color=SERIES[i], lw=1.0, label=a)
        axes[1].hist(d["traded_otm_zone"], bins=30, color=SERIES[i], alpha=0.6, label=a)
    axes[0].set_title("Наблюдаемых OTM-точек в рабочей зоне за день")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
    axes[1].set_title("Распределение по дням")
    axes[1].set_xlabel("точек в день")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "daily_points.png")
    plt.close(fig)


def fig_example_day(b: pd.DataFrame, sync: pd.DataFrame, date: pd.Timestamp, out: Path, asset: str) -> None:
    day = b[(b["date"] == date) & (b["T"] * 365 >= T_MIN_DAYS) & (b["z"].abs() <= 3)]
    s = sync[sync["date"] == date]
    expiries = sorted(day["expiry"].unique())[:4]
    fig, axes = plt.subplots(1, len(expiries), figsize=(3 * len(expiries), 3), sharey=True)
    for ax, e in zip(np.atleast_1d(axes), expiries):
        g = day[day["expiry"] == e].sort_values("strike")
        ax.plot(g["strike"], 100 * g["iv_settle"], color=INK_2, lw=1.2, label="расчётная цена MOEX")
        t = g[g["iv_trade"].notna() & (g["vol_otm"] > 0)]
        ax.scatter(t["strike"], 100 * t["iv_trade"], s=18, facecolors="none", edgecolors=SERIES[1],
                   label="сделки: дневная VWAP")
        ss = s[(s["expiry"] == e) & s["strike"].between(g["strike"].min(), g["strike"].max())]
        ax.scatter(ss["strike"], 100 * ss["iv_sync"], s=16, color=SERIES[0], zorder=3,
                   label="сделки: синхронизация по минутам")
        ax.set_title(f"{e:%d.%m.%Y}, T={g['T'].iloc[0] * 365:.0f} дн.")
        ax.set_xlabel("страйк")
    np.atleast_1d(axes)[0].set_ylabel("IV, %")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper center")
    fig.suptitle(f"{asset}, {date:%d.%m.%Y}: плотная модельная кривая vs реальные сделки", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"example_day_{asset}.png")
    plt.close(fig)


def fig_atm(monthly: dict[str, pd.DataFrame], daily: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for i, a in enumerate(["Si", "RTS", "BR"]):
        if a in monthly and not monthly[a].empty:
            d = monthly[a]["atm30"].replace(0, np.nan)
            axes[0].plot(d.index, 100 * d, color=SERIES[i], label=a)
    for i, (a, d) in enumerate(daily.items()):
        axes[1].plot(d.index, 100 * d["atm30"].replace(0, np.nan), color=SERIES[i], lw=1.0, label=a)
    axes[0].set_title("ATM IV 30 дн., месячные срезы")
    axes[1].set_title("ATM IV 30 дн., дневные данные")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
    for ax in axes:
        ax.set_ylabel("IV, %")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "atm30.png")
    plt.close(fig)


def fig_sync(q: pd.DataFrame, out: Path) -> None:
    bins = [-2, -1, -0.5, 0, 0.5, 1, 2]
    q = q.assign(bucket=pd.cut(q["z"], bins))
    agg = q.groupby("bucket", observed=True).agg(
        naive=("naive_vp", lambda s: np.sqrt(np.mean(s**2))),
        sync=("sync_vp", lambda s: np.sqrt(np.mean(s**2))),
    )
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x - 0.2, agg["naive"], 0.38, color=SERIES[1], label="дневная VWAP и F на клиринге")
    ax.bar(x + 0.2, agg["sync"], 0.38, color=SERIES[0], label="минутная синхронизация с фьючерсом")
    ax.set_xticks(x, [str(i) for i in agg.index])
    ax.set_xlabel("стандартизованная денежность z")
    ax.set_ylabel("RMS отклонения, п.в.")
    ax.set_title("Отклонение IV сделок от IV расчётной цены MOEX")
    ax.set_ylim(0, 1.4 * float(np.nanmax(agg.to_numpy())))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sync_vs_naive.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--monthly-assets", nargs="*", default=["Si", "RTS", "BR", "MIX", "GOLD", "CNY"])
    ap.add_argument("--daily-assets", nargs="*", default=["Si", "RTS"])
    args = ap.parse_args()

    root, out = Path(args.data_dir), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    style()
    summary: dict = {}

    monthly = {}
    for a in args.monthly_assets:
        b = load(root, "monthly", a)
        monthly[a] = daily_density(b) if not b.empty else pd.DataFrame()
        if not b.empty:
            monthly[a].to_csv(out / "tables" / f"density_monthly_{a}.csv")
            m = monthly[a]
            first = m[m["listed"] > 0].index.min()
            by_year = m.groupby(m.index.year)[["listed_zone", "oi_zone", "traded_otm_zone", "series_ge5"]].median()
            by_year.to_csv(out / "tables" / f"density_by_year_{a}.csv")
            summary[f"monthly_{a}"] = {"first_snapshot": str(first.date()), "snapshots": int((m["listed"] > 0).sum())}
    fig_density(monthly, out / "figures")

    daily = {}
    for a in args.daily_assets:
        b = load(root, "daily", a)
        if b.empty:
            continue
        tick = DEFAULT_TICK.get(a, 1.0)
        dd = daily_density(b)
        daily[a] = dd
        dd.to_csv(out / "tables" / f"density_daily_{a}.csv")
        p = persistence(b)
        fit = settle_vs_curve(b)
        fly, cal = arbitrage(b, tick)
        grid = fixed_grid(b)
        grid.to_csv(out / "tables" / f"settle_grid_{a}.csv")
        summary[f"dynamics_{a}"] = surface_dynamics(grid)
        fit.to_csv(out / "tables" / f"settle_curve_fit_{a}.csv", index=False)
        fly.to_csv(out / "tables" / f"butterfly_{a}.csv", index=False)
        cal.to_csv(out / "tables" / f"calendar_{a}.csv", index=False)

        f_check = b.dropna(subset=["F_fut"]).groupby(["date", "expiry"])[["F", "F_fut"]].first()
        rel = (f_check["F"] / f_check["F_fut"] - 1).abs()
        q = dd.describe(percentiles=[0.1, 0.5, 0.9]).T
        summary[f"daily_{a}"] = {
            "days": len(dd),
            "period": f"{dd.index.min().date()}..{dd.index.max().date()}",
            "density_quantiles": q[["10%", "50%", "90%"]].round(2).to_dict(orient="index"),
            "seen_yesterday_median": float(p["seen_yesterday"].median()) if len(p) else None,
            "jaccard_median": float(p["jaccard"].median()) if len(p) else None,
            "settle_curve_rmse_vp_median": float(fit["rmse_vp"].median()) if len(fit) else None,
            "settle_curve_rmse_vp_p90": float(fit["rmse_vp"].quantile(0.9)) if len(fit) else None,
            "butterfly_series_with_viol_share": float((fly["n_viol"] > 0).mean()) if len(fly) else None,
            "butterfly_triplet_viol_share": float(fly["n_viol"].sum() / max(fly["n_triplets"].sum(), 1)),
            "calendar_pairs_with_viol_share": float((cal["share_viol"] > 0).mean()) if len(cal) else None,
            "calendar_bump_vp_median_if_viol": float(cal.loc[cal["max_bump_vp"] > 0.01, "max_bump_vp"].median())
            if len(cal) else None,
            "calendar_bump_vp_p90_if_viol": float(cal.loc[cal["max_bump_vp"] > 0.01, "max_bump_vp"].quantile(0.9))
            if len(cal) else None,
            "calendar_viol_share_T1_le_14d": float((cal.loc[cal["T1"] * 365 <= 14, "share_viol"] > 0).mean())
            if len(cal) else None,
            "calendar_viol_share_T1_gt_14d": float((cal.loc[cal["T1"] * 365 > 14, "share_viol"] > 0).mean())
            if len(cal) else None,
            "forward_parity_vs_futures_rel_err_median": float(rel.median()) if len(rel) else None,
            "forward_parity_vs_futures_rel_err_p99": float(rel.quantile(0.99)) if len(rel) else None,
            "volume_contracts_median": float(dd["volume"].median()),
        }

        sync_path = root / "processed" / f"trades_sync_{a}.csv.gz"
        if sync_path.exists():
            sync = pd.read_csv(sync_path, parse_dates=["date", "expiry"])
            qd = sync_quality(b, sync)
            qd.to_csv(out / "tables" / f"sync_quality_{a}.csv", index=False)
            summary[f"sync_{a}"] = {
                "contract_days": len(qd),
                "rms_naive_vp": float(np.sqrt(np.mean(qd["naive_vp"] ** 2))),
                "rms_sync_vp": float(np.sqrt(np.mean(qd["sync_vp"] ** 2))),
                "mad_naive_vp": float(qd["naive_vp"].abs().median()),
                "mad_sync_vp": float(qd["sync_vp"].abs().median()),
                "intraday_std_vp_median": float(qd["intraday_std_vp"].median()),
            }
            if a == "Si":
                fig_sync(qd, out / "figures")
                counts = qd.groupby("date").size()
                typical = (counts - counts.median()).abs().idxmin()
                fig_example_day(b, sync, typical, out / "figures", a)

    if daily:
        fig_daily_points(daily, out / "figures")
    fig_atm(monthly, daily, out / "figures")
    (out / "tables" / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
