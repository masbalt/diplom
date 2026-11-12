"""TeX-файлы для статьи по сохранённым таблицам основного эксперимента (ничего не пересчитывает).

``paper/generated/exp_numbers.tex`` — числа для текста. Обращение из статьи: ``\\E{Si/rmse/gp_af/random/0.9}``.
Ключи: актив / таблица / ... (см. функцию ``numbers``). Остальные файлы ``exp_tab_*.tex`` — строки таблиц.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pilot import _num

SCHEMES = [("random", 0.1), ("random", 0.3), ("random", 0.5), ("random", 0.7), ("random", 0.9),
           ("wings", None), ("slice", None), ("far", None), ("topn", 40.0), ("topn", 20.0), ("topn", 10.0), ("topn", 5.0)]
MAIN_METHODS = ["cb_quad", "cb_svi", "cb_ssvi", "hist", "hist_cb", "gp_af"]
ABLATION = ["gp", "gp_cb", "gp_indep", "gp_homo", "gp_param", "hist_cb"]
ABLATION_SCHEMES = [("random", 0.1), ("random", 0.5), ("random", 0.9), ("wings", None), ("slice", None),
                    ("far", None), ("topn", 10.0)]
ARB_ORDER = ["cb_quad", "cb_svi", "cb_ssvi", "hist", "hist_cb", "gp", "moex_today"]
ARB_NAMES = {"cb_quad": "Квадратичная улыбка", "cb_svi": "SVI по сериям", "cb_ssvi": "SSVI",
             "hist": "Вчерашняя кривая", "hist_cb": "Вчерашняя кривая + простая поправка",
             "gp": "Предлагаемый метод", "moex_today": "Кривая биржи дня $t$ (справочно)"}
REGIMES = {"calm": "спокойные", "normal": "обычные", "turbulent": "бурные"}
BEST_CB_NAMES = {"cb_quad": "квадратичная улыбка", "cb_svi": "SVI", "cb_ssvi": "SSVI"}


def scheme_label(scheme: str, m) -> str:
    if scheme == "random":
        return f"Случайно, {100 * m:.0f}\\,\\%"
    if scheme == "topn":
        return f"Видны {m:.0f} самых торгуемых"
    return {"wings": "Крылья: видны только $|z|\\le 1$", "slice": "Скрыта целая серия",
            "far": "Видны две ближние серии"}[scheme]


def key(scheme: str, m) -> str:
    return scheme if m is None else f"{scheme}/{m:g}"


def ascii_key(label) -> str:
    """Подпись корзины без не-ASCII символов: в имени TeX-макроса они недопустимы."""
    return str(label).replace("–", "-").replace(",", ".")


def signed(x: float, digits: int = 2) -> str:
    return ("\\ensuremath{-}" if x < 0 else "") + _num(abs(x), digits)


def _pick(t: pd.DataFrame, scheme: str, m) -> pd.DataFrame:
    r = t[t["scheme"] == scheme]
    return r[r["mask"].isna()] if m is None else r[np.isclose(r["mask"], m)]


def _schemes(t: pd.DataFrame, schemes=SCHEMES):
    for scheme, m in schemes:
        r = _pick(t, scheme, m)
        if len(r):
            yield scheme, m, r


# --- числа для текста ----------------------------------------------------------------

def numbers(tables: Path, asset: str) -> dict[str, str]:
    out: dict[str, str] = {}
    info = json.loads((tables / f"exp_info_{asset}.json").read_text())
    for part in ("train", "test"):
        out[f"info/{part}_days"] = str(info[f"{part}_days"])
        out[f"info/{part}_points"] = _num(info[f"{part}_points"], 0)
        a, b = info[f"{part}_period"].split("..")
        out[f"info/{part}_period"] = f"{a}\\,--\\,{b}"
    out["info/points_per_day"] = _num(info["points_per_day_median"], 0)
    out["info/series_per_day"] = _num(info["series_per_day_median"], 0)
    out["info/gross_removed"] = _num(info["gross_removed"], 0)
    gap = info.get("series_gap")
    if gap:
        out["gap/total"], out["gap/worst"] = _num(gap["total"], 0), str(gap["worst"])
        out["gap/hist_rmse"], out["gap/hist_rmse_rest"] = _num(gap["hist_rmse"], 1), _num(gap["hist_rmse_rest"], 1)

    rmse = pd.read_csv(tables / f"exp_rmse_{asset}.csv")
    for scheme, m, r in _schemes(rmse):
        for row in r.itertuples():
            base = f"{row.method}/{key(scheme, m)}"
            out[f"rmse/{base}"] = _num(row.rmse)
            out[f"mae/{base}"] = _num(row.mae)
            out[f"price/{base}"] = _num(row.price_err_pct, 1)
        out[f"visible/{key(scheme, m)}"] = _num(r["visible_median"].iloc[0], 0)
        out[f"bestcb/{key(scheme, m)}"] = BEST_CB_NAMES[r[r["method"] == "best_cb"]["source"].iloc[0]]
    diff = pd.read_csv(tables / f"exp_diff_{asset}.csv")
    for scheme, m, r in _schemes(diff):
        for row in r.itertuples():
            base = f"{row.a}/{row.b}/{key(scheme, m)}"
            out[f"diff/{base}"] = signed(row.diff)
            out[f"ci/{base}"] = f"[{signed(row.lo)};\\ {signed(row.hi)}]"
    for name in ("visible", "T", "z", "n"):
        t = pd.read_csv(tables / f"exp_by_{name}_{asset}.csv")
        for row in t.to_dict("records"):
            for meth, v in row.items():
                if meth not in ("bucket", "n"):
                    out[f"by_{name}/{meth}/{ascii_key(row['bucket'])}"] = _num(v)
    reg = pd.read_csv(tables / f"exp_regimes_{asset}.csv")
    for row in reg.to_dict("records"):
        for meth, v in row.items():
            if meth not in ("mask", "regime", "days"):
                out[f"regime/{meth}/{row['mask']:g}/{row['regime']}"] = _num(v)
    cal = pd.read_csv(tables / f"exp_calibration_{asset}.csv")
    for scheme, m, r in _schemes(cal):
        for q in (50, 80, 95):
            out[f"cover/{q}/{key(scheme, m)}"] = _num(100 * r[f"cover_{q}"].iloc[0], 0)
    lag_path = tables / f"exp_lags_{asset}.csv"
    if lag_path.exists():
        for row in pd.read_csv(lag_path).itertuples():
            out[f"lag/{row.method}/{row.mask:g}/{row.lag}"] = _num(row.rmse)
    arb_path = tables / f"exp_arbitrage_{asset}.csv"
    if arb_path.exists():
        for row in pd.read_csv(arb_path).to_dict("records"):
            base = f"{row['scenario']}/{row['method']}"
            for col in ("butterfly_days", "calendar_days", "any_days"):
                out[f"arb/{col}/{base}"] = _num(100 * row[col], 0)
            out[f"arb/repair_mean/{base}"] = _num(row["repair_mean_vp"], 3)
            if row["scenario"] == "sparse":
                out[f"arb/rmse_before/{base}"] = _num(row["rmse_before"])
                out[f"arb/rmse_after/{base}"] = _num(row["rmse_after"])
                out[f"arb/zone_share/{base}"] = _num(100 * row["zone_share"], 0)
    P = json.loads((tables / f"gp_params_{asset}.json").read_text())
    for variant in ("gp", "gp_cb"):
        for name, v in P[variant].items():
            if name in ("ell_T", "ell_x"):
                out[f"param/{variant}/{name}"] = _num(v)
            elif name not in ("nll", "indep"):
                out[f"param/{variant}/{name}"] = _num(100 * v)   # масштабы в п.п. IV
    return out


# --- строки таблиц ------------------------------------------------------------------

def _row(cells: list[str]) -> str:
    return " & ".join(cells) + " \\\\"


def _best_bold(values: list[float], digits: int = 2) -> list[str]:
    best = min(values)
    return [f"\\textbf{{{_num(v, digits)}}}" if round(v, digits) == round(best, digits) else _num(v, digits) for v in values]


def tab_main(tables: Path, asset: str, column: str = "rmse", digits: int = 2) -> list[str]:
    """Схема | видимых сделок | методы | кривая биржи дня t. Жирным — лучший метод в строке."""
    t = pd.read_csv(tables / f"exp_rmse_{asset}.csv")
    lines, last = [], None
    for scheme, m, r in _schemes(t):
        if last is not None and scheme != last:
            lines.append("\\midrule")
        last = scheme
        v = r.set_index("method")[column]
        lines.append(_row([scheme_label(scheme, m), _num(r["visible_median"].iloc[0], 0),
                           *_best_bold([v[k] for k in MAIN_METHODS], digits), _num(v["moex_today"], digits)]))
    return lines


def tab_diff(tables: Path, asset: str) -> list[str]:
    """Разности RMSE предлагаемого метода с базовыми и 95 % интервалы."""
    t = pd.read_csv(tables / f"exp_diff_{asset}.csv")
    lines, last = [], None
    for scheme, m, r in _schemes(t):
        if last is not None and scheme != last:
            lines.append("\\midrule")
        last = scheme
        cells = [scheme_label(scheme, m)]
        for b in ("best_cb", "hist", "hist_cb"):
            d = r[(r["a"] == "gp_af") & (r["b"] == b)].iloc[0]
            cells.append(f"{signed(d['diff'])} [{signed(d['lo'])};\\ {signed(d['hi'])}]")
        lines.append(_row(cells))
    return lines


def tab_ablation(tables: Path, asset: str) -> list[str]:
    t = pd.read_csv(tables / f"exp_rmse_{asset}.csv")
    lines = []
    for scheme, m, r in _schemes(t, ABLATION_SCHEMES):
        v = r.set_index("method")["rmse"]
        lines.append(_row([scheme_label(scheme, m), *_best_bold([v[k] for k in ABLATION])]))
    return lines


def tab_arbitrage(tables: Path, asset: str) -> list[str]:
    t = pd.read_csv(tables / f"exp_arbitrage_{asset}.csv").set_index(["scenario", "method"])
    lines = []
    for meth in ARB_ORDER:
        f, s = t.loc[("full", meth)], t.loc[("sparse", meth)]
        if meth == "moex_today":
            lines.append("\\midrule")
        lines.append(_row([ARB_NAMES[meth], _num(100 * f["butterfly_days"], 0), _num(100 * f["calendar_days"], 0),
                           _num(f["repair_mean_vp"], 3), _num(100 * s["butterfly_days"], 0),
                           _num(100 * s["calendar_days"], 0), _num(s["repair_mean_vp"], 3),
                           _num(s["rmse_before"]), _num(s["rmse_after"])]))
    return lines


def tab_regimes(tables: Path, asset: str, masks=(0.1, 0.5, 0.9)) -> list[str]:
    t = pd.read_csv(tables / f"exp_regimes_{asset}.csv")
    methods = ["cb_ssvi", "hist", "hist_cb", "gp_af", "moex_today"]
    lines = []
    for i, m in enumerate(masks):
        if i:
            lines.append("\\midrule")
        for row in t[np.isclose(t["mask"], m)].to_dict("records"):
            vals = [row[k] for k in methods]
            lines.append(_row([f"{100 * m:.0f}\\,\\%", REGIMES[row["regime"]], _num(row["move_median"]),
                               *_best_bold(vals[:-1]), _num(vals[-1])]))
    return lines


def tab_calibration(tables: Path, asset: str) -> list[str]:
    t = pd.read_csv(tables / f"exp_calibration_{asset}.csv")
    lines = []
    for scheme, m, r in _schemes(t):
        r = r.iloc[0]
        lines.append(_row([scheme_label(scheme, m), _num(r["sd_median_vp"]),
                           *[_num(100 * r[f"cover_{q}"], 0) for q in (50, 80, 95)]]))
    return lines


def tab_params(tables: Path, asset: str) -> list[str]:
    P = json.loads((tables / f"gp_params_{asset}.json").read_text())
    rows = [("Общий сдвиг: уровень $g_0$, наклон $g_1$, кривизна $g_2$", ["g0", "g1", "g2"]),
            ("Сдвиг, затухающий по сроку: $t_0$, $t_1$, $t_2$", ["t0", "t1", "t2"]),
            ("Сдвиг серии $t_s$ и локальная часть $t_{loc}$", ["ts", "tloc"]),
            ("Шум сделки: $a$, $b$", ["a", "b"])]
    lines = []
    for label, names in rows:
        cells = [label]
        for variant in ("gp", "gp_cb"):
            cells.append("; ".join(_num(100 * P[variant][k]) for k in names))
        lines.append(_row(cells))
    lines.append(_row(["Длина связи по сроку $\\ell_T$ (в единицах $\\ln T$)", _num(P["gp"]["ell_T"]), _num(P["gp_cb"]["ell_T"])]))
    lines.append(_row(["Длина связи по денежности $\\ell_x$", _num(P["gp"]["ell_x"]), _num(P["gp_cb"]["ell_x"])]))
    return lines


def write(tables: Path, out: Path, asset: str, extra_assets: list[str]) -> None:
    """Пишет exp_numbers.tex и строки таблиц для основного актива и для активов из ``extra_assets``,
    если по ним есть результаты."""
    out.mkdir(parents=True, exist_ok=True)
    assets = [asset] + [a for a in extra_assets if (tables / f"exp_rmse_{a}.csv").exists()]
    lines = ["% Сгенерировано code/main.py (этап figures) — не редактировать вручную.",
             "\\newcommand{\\E}[1]{\\csname E/#1\\endcsname}"]
    for a in assets:
        for k, v in numbers(tables, a).items():
            lines.append(f"\\expandafter\\def\\csname E/{a}/{k}\\endcsname{{{v}}}")
    (out / "exp_numbers.tex").write_text("\n".join(lines) + "\n")
    for a in assets:
        files = {"main": tab_main(tables, a), "mae": tab_main(tables, a, "mae"),
                 "price": tab_main(tables, a, "price_err_pct", 1), "diff": tab_diff(tables, a),
                 "ablation": tab_ablation(tables, a), "regimes": tab_regimes(tables, a),
                 "calibration": tab_calibration(tables, a), "params": tab_params(tables, a)}
        if (tables / f"exp_arbitrage_{a}.csv").exists():
            files["arbitrage"] = tab_arbitrage(tables, a)
        for name, rows in files.items():
            (out / f"exp_tab_{name}_{a}.tex").write_text("\n".join(rows) + "\n")
