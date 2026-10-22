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
