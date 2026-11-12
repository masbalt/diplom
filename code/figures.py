"""Рисунки проекта. Только визуализация: читает сохранённые таблицы, ничего не пересчитывает.

Шрифт — Nimbus Roman (клон Times с кириллицей), как Tempora в статье; формулы — STIX.
Рисунки сохраняются в PDF (вектор, для статьи и слайдов) и PNG (для README/отчёта).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # синий, оранжевый, бирюзовый — фиксированный порядок

LABELS = {
    "hist": "только история (вчерашняя кривая)",
    "cb_quad": "только текущая доска: квадратичная улыбка",
    "cb_svi": "только текущая доска: SVI",
    "hist_cb": "история + текущая доска",
    "moex_today": "кривая MOEX дня D (полный стакан, справочно)",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
        "legend.frameon": False, "figure.dpi": 150,
    })


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def pilot_main(tables: Path, out: Path, asset: str) -> None:
    """Главный рисунок: ошибка восстановления скрытых сделок от доли скрытых сделок.

    Методы: лучший «только текущие сделки», «только вчерашняя кривая», «вчерашняя кривая + текущие сделки».
    Полосы — 95 % бутстреп-интервалы по дням. Размер подобран под слайд 16:9 (≈11.5 см в ширину).
    """
    c = pd.read_csv(tables / f"pilot_curve_{asset}.csv")
    info = json.loads((tables / f"pilot_info_{asset}.json").read_text())
    lines = [
        ("best_cb", SERIES[1], "-", r"только текущие сделки  $f(\mathcal{O}^{\mathrm{vis}}_t)$"),
        ("hist", SERIES[2], "--", r"только вчерашняя кривая  $S_{t-1}$"),
        ("hist_cb", SERIES[0], "-", r"вчерашняя кривая + текущие сделки  $f(\mathcal{O}^{\mathrm{vis}}_t, S_{t-1})$"),
    ]
    fig, ax = plt.subplots(figsize=(4.55, 2.95))
    for meth, color, ls, label in lines:
        d = c[c["method"] == meth].sort_values("mask")
        x = 100 * d["mask"]
        ax.fill_between(x, d["lo"], d["hi"], color=color, alpha=0.16, lw=0)
        ax.plot(x, d["rmse"], color=color, ls=ls, marker="o", ms=3.5, label=label, lw=1.7 if ls == "-" else 1.1)

    # Значения при наибольшей доле скрытия — главный результат.
    for meth, color in [("best_cb", SERIES[1]), ("hist_cb", SERIES[0])]:
        d = c[c["method"] == meth].set_index("mask")
        hi_m = d.index.max()
        ax.annotate(f"{d.loc[hi_m, 'rmse']:.2f}".replace(".", ","), (100 * hi_m, d.loc[hi_m, "rmse"]),
                    xytext=(5, 0), textcoords="offset points", va="center", color=color, fontsize=9,
                    fontweight="bold")
    # При наименьшей доле скрытия — подпись, только если различие статистически незначимо.
    lo_m = c["mask"].min()
    _, ci_lo, ci_hi = info["comparisons_rmse_diff_vp"][f"random_{lo_m}"]["hist_cb_minus_best_cb"]
    if ci_lo < 0 < ci_hi:
        y = c[(c["mask"] == lo_m) & c["method"].isin(["best_cb", "hist_cb"])]["rmse"].max()
        ax.annotate(f"при {100 * lo_m:.0f} %: различие незначимо", (100 * lo_m, y), xytext=(2, 16),
                    textcoords="offset points", fontsize=7.5, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.5))

    vis = c[c["method"] == "hist_cb"].sort_values("mask")
    ax.set_xticks(100 * vis["mask"],
                  [f"{100 * m:.0f} %\n≈{v:.0f}" for m, v in zip(vis["mask"], vis["visible_median"])])
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(4, 100)
    ax.set_ylim(0, None)
    ax.set_xlabel("доля скрытых сделок дня $m$ (снизу — видимых сделок в день)", fontsize=8)
    ax.set_ylabel("RMSE в скрытых сделках, п.п. IV", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, handlelength=2.0, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, out, f"oslide_pilot_{asset}")


def pilot_all_methods(tables: Path, out: Path, asset: str) -> None:
    """Все методы, схема random (рисунок для отчёта и статьи)."""
    c = pd.read_csv(tables / f"pilot_curve_{asset}.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    spec = [("cb_svi", SERIES[1], "-"), ("cb_quad", SERIES[1], "--"), ("hist", SERIES[2], "-"),
            ("hist_cb", SERIES[0], "-"), ("moex_today", INK_2, ":")]
    for meth, color, ls in spec:
        d = c[c["method"] == meth].sort_values("mask")
        ax.plot(100 * d["mask"], d["rmse"], color=color, ls=ls, marker=None if ls == ":" else "o", ms=4,
                lw=1.2 if ls != "-" else 1.6, label=LABELS[meth])
    ax.set_xlabel("доля скрытых сделок дня, %")
    ax.set_ylabel("RMSE в скрытых точках, п.п. IV")
    ax.set_title(f"{asset}: восстановление скрытых сделок", color=INK)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, f"pilot_masking_{asset}")


def pilot_figures(tables: Path, out: Path, asset: str) -> None:
    style()
    pilot_main(tables, out, asset)
    pilot_all_methods(tables, out, asset)


# --- основной эксперимент -------------------------------------------------------
# Цвет закреплён за методом и один и тот же на всех рисунках. Метод без истории (gp_cb) —
# тот же синий, что у предлагаемого метода, но пунктиром: это тот же метод без вчерашней кривой.
EXP_STYLE = {
    "gp_af": ("#2a78d6", "-", "o", "предлагаемый метод"),
    "gp": ("#2a78d6", "-", "o", "предлагаемый метод"),
    "cb_ssvi": ("#eb6834", "-", "s", "только текущие сделки: SSVI"),
    "hist": ("#1baf7a", "-", "^", "только вчерашняя кривая"),
    "hist_cb": ("#eda100", "-", "D", "вчерашняя кривая + простая поправка"),
    "cb_quad": ("#e87ba4", "-", "v", "только текущие сделки: квадратичная улыбка"),
    "gp_cb": ("#2a78d6", "--", None, "предлагаемый метод без вчерашней кривой"),
    "moex_today": (INK_2, ":", None, "кривая биржи дня $t$ (справочно)"),
}
EXP_LINES = ["cb_quad", "cb_ssvi", "hist", "hist_cb", "gp_af", "moex_today"]
TICKS_Y = [0.6, 0.8, 1, 1.5, 2, 3, 4, 5]
# На рисунке об устаревании истории кривая имеет возраст h дней, поэтому подписи другие.
LAG_LABELS = {"hist": "только прошлая кривая", "hist_cb": "прошлая кривая + простая поправка",
              "gp": "предлагаемый метод с прошлой кривой"}


def _line(ax, x, y, meth: str, lo=None, hi=None, label: str | None = None) -> None:
    color, ls, marker, name = EXP_STYLE[meth]
    if lo is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)
    ax.plot(x, y, color=color, ls=ls, marker=marker, ms=4, lw=2.0 if ls == "-" else 1.3,
            label=label or name, solid_capstyle="round")


def _log_axis(ax, lo: float, hi: float) -> None:
    ax.set_yscale("log")
    ticks = [t for t in TICKS_Y if lo <= t <= hi]
    ax.set_yticks(ticks, [str(t).replace(".", ",") for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(lo, hi)


def exp_main(tables: Path, out: Path, asset: str) -> None:
    """Главный рисунок: ошибка восстановления при случайном скрытии и когда видны только
    N самых торгуемых контрактов. Полосы — 95 % бутстреп-интервалы по дням."""
    r = pd.read_csv(tables / f"exp_rmse_{asset}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.1), sharey=True)
    for ax, scheme in zip(axes, ["random", "topn"]):
        d = r[r["scheme"] == scheme]
        vis = d.drop_duplicates("mask").sort_values("mask")
        pos = {m: i for i, m in enumerate(vis["mask"])}
        for meth in EXP_LINES:
            g = d[d["method"] == meth].sort_values("mask")
            band = meth != "moex_today"
            _line(ax, g["mask"].map(pos), g["rmse"], meth, g["lo"] if band else None, g["hi"] if band else None)
        if scheme == "random":
            ax.set_xticks(list(pos.values()), [f"{100 * m:.0f} %\n≈{v:.0f}" for m, v in zip(vis["mask"], vis["visible_median"])])
            ax.set_xlabel("доля скрытых сделок дня\n(снизу — видимых сделок в день)")
            ax.set_ylabel("RMSE в скрытых сделках, п.п. IV")
            ax.set_title("а) случайное скрытие", loc="left")
        else:
            ax.set_xticks(list(pos.values()), [f"{m:.0f}" for m in vis["mask"]])
            ax.invert_xaxis()
            ax.set_xlabel("видно $N$ самых торгуемых\nконтрактов дня")
            ax.set_title("б) видны только самые торгуемые", loc="left")
        ax.tick_params(labelsize=8)
    _log_axis(axes[0], 0.7, 4.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, handlelength=2.2, columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.19, 1, 1))
    save(fig, out, f"exp_main_{asset}")


def exp_breakdowns(tables: Path, out: Path, asset: str) -> None:
    """Ошибка по числу видимых сделок в той же серии, по сроку и по денежности."""
    spec = [("visible", "видимых сделок\nв той же серии", "а) видно сделок серии"),
            ("T", "дней до экспирации", "б) срок серии"),
            ("z", r"денежность $|z|$", "в) денежность")]
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.9), sharey=True)
    for ax, (name, xlabel, title) in zip(axes, spec):
        t = pd.read_csv(tables / f"exp_by_{name}_{asset}.csv")
        x = range(len(t))
        for meth in EXP_LINES:
            _line(ax, x, t[meth], meth)
        ax.set_xticks(list(x), t["bucket"].astype(str), rotation=30 if name == "T" else 0)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("RMSE, п.п. IV")
    _log_axis(axes[0], 0.6, 5.6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, handlelength=2.2, columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.2, 1, 1))
    save(fig, out, f"exp_breakdowns_{asset}")


def exp_lags(tables: Path, out: Path, asset: str) -> None:
    """Устаревание истории: ошибка при вчерашней кривой возрастом h торговых дней."""
    t = pd.read_csv(tables / f"exp_lags_{asset}.csv")
    masks = sorted(t["mask"].unique())
    fig, axes = plt.subplots(1, len(masks), figsize=(5.6, 3.0), sharey=True)
    for ax, m, letter in zip(axes, masks, "абвг"):
        d = t[t["mask"] == m]
        lags = sorted(d["lag"].unique())
        pos = {h: i for i, h in enumerate(lags)}
        for meth in ["hist", "hist_cb", "gp"]:
            g = d[d["method"] == meth].sort_values("lag")
            _line(ax, g["lag"].map(pos), g["rmse"], meth, g["lo"], g["hi"], LAG_LABELS[meth])
        for meth in ["gp_cb", "cb_ssvi"]:   # не зависят от возраста истории
            color, ls, _, name = EXP_STYLE[meth]
            ax.axhline(d[d["method"] == meth]["rmse"].iloc[0], color=color, ls="--", lw=1.3, label=name)
        ax.set_xticks(list(pos.values()), [str(h) for h in lags])
        ax.set_xlabel("возраст истории, торговых дней")
        ax.set_title(f"{letter}) скрыто {100 * m:.0f} % сделок", loc="left")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("RMSE в скрытых сделках, п.п. IV")
    _log_axis(axes[0], 0.7, 4.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, handlelength=2.2, columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0, 0.2, 1, 1))
    save(fig, out, f"exp_lags_{asset}")


def exp_example(tables: Path, out: Path, asset: str) -> None:
    """Пример одного дня: видимые и скрытые сделки трёх самых торгуемых серий и восстановленные кривые."""
    t = pd.read_csv(tables / f"exp_example_{asset}.csv", parse_dates=["date", "expiry"])
    pts = t[t["kind"] != "curve"]
    top = pts.groupby("expiry").size().nlargest(3).index.sort_values()
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.6))
    for ax, e in zip(axes, top):
        c = t[(t["kind"] == "curve") & (t["expiry"] == e)]
        k = 100 * c["k"]
        ax.fill_between(k, 100 * (c["gp"] - 2 * c["gp_sd"]), 100 * (c["gp"] + 2 * c["gp_sd"]),
                        color=EXP_STYLE["gp"][0], alpha=0.14, lw=0)
        for meth in ["cb_ssvi", "hist", "gp"]:
            color, ls, _, name = EXP_STYLE[meth]
            ax.plot(k, 100 * c[meth], color=color, lw=1.8, label=name)
        for kind, face, name in [("hidden", SURFACE, "скрытые сделки"), ("visible", INK, "видимые сделки")]:
            p = pts[(pts["kind"] == kind) & (pts["expiry"] == e)]
            ax.scatter(100 * p["k"], 100 * p["iv"], s=16, facecolor=face, edgecolor=INK, linewidth=0.8, zorder=3,
                       label=name)
        lo, hi = 100 * pts[pts["expiry"] == e]["iv"].min(), 100 * pts[pts["expiry"] == e]["iv"].max()
        ax.set_ylim(lo - 0.25 * (hi - lo) - 1, hi + 0.25 * (hi - lo) + 1)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, integer=True))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, integer=True))
        ax.set_title(f"{round(365 * c['T'].iloc[0])} дн. до экспирации", loc="left", fontsize=9)
        ax.set_xlabel(r"$k=\ln(K/F)$, %")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("IV, %")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.4, rect=(0.01, 0.16, 1, 1))
    save(fig, out, f"exp_example_{asset}")


def experiment_figures(tables: Path, out: Path, asset: str) -> None:
    style()
    exp_main(tables, out, asset)
    exp_breakdowns(tables, out, asset)
    exp_lags(tables, out, asset)
    if (tables / f"exp_example_{asset}.csv").exists():
        exp_example(tables, out, asset)
