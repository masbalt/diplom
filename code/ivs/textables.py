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
ARB_ORDER = ["cb_quad", "cb_svi", "cb_ssvi", "hist", "hist_cb", "gp", "gp_dyn", "moex_today"]
ARB_NAMES = {"cb_quad": "Квадратичная улыбка", "cb_svi": "SVI по сериям", "cb_ssvi": "SSVI",
             "hist": "Вчерашняя кривая", "hist_cb": "Вчерашняя кривая + простая поправка",
             "gp": "Предлагаемый метод", "gp_dyn": "Предлагаемый метод, динамический вариант",
             "moex_today": "Кривая биржи дня $t$ (справочно)"}
FORECAST = ["fc_k", "fc_strike", "fc_x", "fc_ar", "fc_gbm", "fc_carry"]
DYN_METHODS = ["hist", "dyn", "gp_af", "gp_dyn_af"]
NEURAL_PAIRS = [("gp", "np"), ("gp_cb", "np_cb"), ("gp_dyn", "np_dyn")]
HOLDOUT_METHODS = ["hist", "gp_af", "gp_dyn_af"]
QUOTE_NAMES = {"ssvi": "SSVI по сделкам дня $t$", "exchange": "Расчётная кривая биржи дня $t$",
               "gp_dyn": "Предлагаемый метод по сделкам дня $t$"}
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

def _rmse_numbers(out: dict, tables: Path, tag: str, prefix: str = "") -> None:
    """RMSE и разности методов по сценариям; ``prefix`` — приставка ключей (``H/`` для контрольного периода)."""
    rmse = pd.read_csv(tables / f"exp_rmse_{tag}.csv")
    for scheme, m, r in _schemes(rmse):
        for row in r.itertuples():
            base = f"{row.method}/{key(scheme, m)}"
            out[f"{prefix}rmse/{base}"] = _num(row.rmse)
            out[f"{prefix}mae/{base}"] = _num(row.mae)
            out[f"{prefix}price/{base}"] = _num(row.price_err_pct, 1)
        out[f"{prefix}visible/{key(scheme, m)}"] = _num(r["visible_median"].iloc[0], 0)
        out[f"{prefix}bestcb/{key(scheme, m)}"] = BEST_CB_NAMES[r[r["method"] == "best_cb"]["source"].iloc[0]]
    diff = pd.read_csv(tables / f"exp_diff_{tag}.csv")
    for scheme, m, r in _schemes(diff):
        for row in r.itertuples():
            base = f"{row.a}/{row.b}/{key(scheme, m)}"
            out[f"{prefix}diff/{base}"] = signed(row.diff)
            out[f"{prefix}ci/{base}"] = f"[{signed(row.lo)};\\ {signed(row.hi)}]"
            out[f"{prefix}diff3/{base}"] = signed(row.diff, 3)   # три знака: для разностей в тысячных долях п.п.
            out[f"{prefix}ci3/{base}"] = f"[{signed(row.lo, 3)};\\ {signed(row.hi, 3)}]"


def _forecast_numbers(out: dict, tables: Path, tag: str, prefix: str = "") -> None:
    """Ошибка прогноза по горизонтам: ``fc/метод/h`` и разности ``fcdiff/a/b/h`` с интервалами ``fcci``."""
    for row in pd.read_csv(tables / f"exp_forecast_{tag}.csv").itertuples():
        out[f"{prefix}fc/{row.method}/{row.h}"] = _num(row.rmse)
    for row in pd.read_csv(tables / f"exp_forecast_diff_{tag}.csv").itertuples():
        out[f"{prefix}fcdiff/{row.a}/{row.b}/{row.h}"] = signed(row.diff)
        out[f"{prefix}fcci/{row.a}/{row.b}/{row.h}"] = f"[{signed(row.lo)};\\ {signed(row.hi)}]"


def numbers(tables: Path, asset: str) -> dict[str, str]:
    out: dict[str, str] = {}
    info = json.loads((tables / f"exp_info_{asset}.json").read_text())
    for part in ("train", "test", "holdout"):
        if f"{part}_days" in info:
            out[f"info/{part}_days"] = str(info[f"{part}_days"])
            out[f"info/{part}_points"] = _num(info[f"{part}_points"], 0)
            a, b = info[f"{part}_period"].split("..")
            out[f"info/{part}_period"] = f"{a}\\,--\\,{b}"
    out["info/points_per_day"] = _num(info["points_per_day_median"], 0)
    out["info/series_per_day"] = _num(info["series_per_day_median"], 0)
    out["info/gross_removed"] = _num(info["gross_removed"], 0)
    for name, prefix in (("series_gap", "gap"), ("series_gap_dyn", "gap_dyn")):
        gap = info.get(name)
        if gap:
            out[f"{prefix}/total"], out[f"{prefix}/worst"] = _num(gap["total"], 0), str(gap["worst"])
            if gap["worst"]:
                out[f"{prefix}/hist_rmse"] = _num(gap["hist_rmse"], 1)
            out[f"{prefix}/hist_rmse_rest"] = _num(gap["hist_rmse_rest"], 1)

    _rmse_numbers(out, tables, asset)
    if (tables / f"exp_rmse_holdout_{asset}.csv").exists():
        _rmse_numbers(out, tables, f"holdout_{asset}", "H/")
        _rmse_numbers(out, tables, f"holdout_trim_{asset}", "HT/")   # без одной пары «серия, день»
        worst = json.loads((tables / f"exp_holdout_outlier_{asset}.json").read_text())
        for name in ("date", "expiry"):
            y, m, d = worst[name].split("-")
            out[f"houtlier/{name}"] = f"{d}.{m}.{y}"
        out["houtlier/trades"], out["houtlier/pairs"] = str(worst["trades"]), str(worst["pairs"])
        out["houtlier/err"], out["houtlier/share"] = _num(worst["hist_err_vp"], 1), _num(100 * worst["share_sse"], 0)
    if (tables / f"exp_forecast_{asset}.csv").exists():
        _forecast_numbers(out, tables, asset)
        for row in pd.read_csv(tables / f"exp_forecast_buckets_{asset}.csv").to_dict("records"):
            for meth in FORECAST:
                out[f"fcb/{row['by']}/{row['h']}/{meth}/{ascii_key(row['bucket'])}"] = _num(row[meth])
    if (tables / f"exp_forecast_holdout_{asset}.csv").exists():
        _forecast_numbers(out, tables, f"holdout_{asset}", "H/")
    quotes = tables / f"exp_quotes_{asset}.csv"
    if quotes.exists():
        q = pd.read_csv(quotes)
        out["quotes/day"], out["quotes/time"] = q["day"].iloc[0], q["time"].iloc[0][11:16]
        for row in q.itertuples():
            out[f"quotes/{row.subset}/n"], out[f"quotes/{row.subset}/series"] = str(row.n), str(row.series)
            out[f"quotes/{row.subset}/spread"] = _num(row.spread_median_vp, 1)
            out[f"quotes/{row.subset}/{row.method}/inside"] = _num(100 * row.inside, 0)
            out[f"quotes/{row.subset}/{row.method}/mad"] = _num(row.mad_vp)
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
    for variant in ("gp", "gp_cb", "gp_dyn"):
        for name, v in P.get(variant, {}).items():
            if name in ("ell_T", "ell_x"):
                out[f"param/{variant}/{name}"] = _num(v)
            elif name not in ("nll", "indep"):
                out[f"param/{variant}/{name}"] = _num(100 * v)   # масштабы в п.п. IV
    fc_path = tables / f"exp_forecast_params_{asset}.json"
    if fc_path.exists():
        fc = json.loads(fc_path.read_text())
        for h, phi in fc["phi"].items():
            out[f"phi/{h}"] = _num(phi)
            out[f"phicurve/{h}"] = _num(100 * fc["phi_curve"][h], 0)   # в процентах расхождения
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
        if ("full", meth) not in t.index:
            continue
        f, s = t.loc[("full", meth)], t.loc[("sparse", meth)]
        if meth == "moex_today":
            lines.append("\\midrule")
        lines.append(_row([ARB_NAMES[meth], _num(100 * f["butterfly_days"], 0), _num(100 * f["calendar_days"], 0),
                           _num(f["repair_mean_vp"], 3), _num(100 * s["butterfly_days"], 0),
                           _num(100 * s["calendar_days"], 0), _num(s["repair_mean_vp"], 3),
                           _num(s["rmse_before"]), _num(s["rmse_after"])]))
    return lines


def _ci(d, digits: int = 2) -> str:
    return f"{signed(d['diff'], digits)} [{signed(d['lo'], digits)};\\ {signed(d['hi'], digits)}]"


def tab_forecast(tables: Path, tag: str) -> list[str]:
    """Горизонт | RMSE шести прогнозов | разность «перенос поправки − перенос в k» с интервалом."""
    t = pd.read_csv(tables / f"exp_forecast_{tag}.csv")
    d = pd.read_csv(tables / f"exp_forecast_diff_{tag}.csv")
    lines = []
    for h in sorted(t["h"].unique()):
        v = t[t["h"] == h].set_index("method")["rmse"]
        diff = d[(d["h"] == h) & (d["a"] == "fc_carry") & (d["b"] == "fc_k")].iloc[0]
        lines.append(_row([str(h), *_best_bold([v[k] for k in FORECAST]), _ci(diff)]))
    return lines


def _scheme_rows(tables: Path, tag: str, cells) -> list[str]:
    """Строки по сценариям скрытия; ``cells(v, scheme, m)`` — ячейки после названия сценария и числа видимых."""
    t = pd.read_csv(tables / f"exp_rmse_{tag}.csv")
    lines, last = [], None
    for scheme, m, r in _schemes(t):
        if last is not None and scheme != last:
            lines.append("\\midrule")
        last = scheme
        lines.append(_row([scheme_label(scheme, m), _num(r["visible_median"].iloc[0], 0),
                           *cells(r.set_index("method")["rmse"], scheme, m)]))
    return lines


def tab_dyn(tables: Path, asset: str) -> list[str]:
    """Динамический вариант: начальные приближения, базовый метод, варианты и разность с базовым методом."""
    d = pd.read_csv(tables / f"exp_diff_{asset}.csv")
    d = d[(d["a"] == "gp_dyn_af") & (d["b"] == "gp_af")]
    return _scheme_rows(tables, asset, lambda v, scheme, m: [*_best_bold([v[k] for k in DYN_METHODS]),
                                                               _ci(_pick(d, scheme, m).iloc[0], 3)])


def tab_neural(tables: Path, asset: str) -> list[str]:
    """Гауссовский процесс и нейросеть при одном и том же начальном приближении; жирным — лучший в паре."""
    return _scheme_rows(tables, asset, lambda v, scheme, m: [c for pair in NEURAL_PAIRS
                                                               for c in _best_bold([v[k] for k in pair])])


def tab_holdout(tables: Path, asset: str) -> list[str]:
    """Контрольный период. Столбцы: лучший метод текущей доски; методы с историей по всем сделкам;
    те же методы без одной пары «серия, день» с наибольшей ошибкой вчерашней кривой; кривая биржи дня t.
    Жирным — лучший из четырёх методов в каждой из двух групп."""
    trim = pd.read_csv(tables / f"exp_rmse_holdout_trim_{asset}.csv")

    def cells(v, scheme, m):
        w = _pick(trim, scheme, m).set_index("method")["rmse"]
        full = _best_bold([v["best_cb"], *[v[k] for k in HOLDOUT_METHODS]])
        cut = _best_bold([w["best_cb"], *[w[k] for k in HOLDOUT_METHODS]])
        return [*full, *cut[1:], _num(v["moex_today"])]

    return _scheme_rows(tables, f"holdout_{asset}", cells)


def tab_quotes(tables: Path, asset: str) -> list[str]:
    t = pd.read_csv(tables / f"exp_quotes_{asset}.csv").set_index(["subset", "method"])
    return [_row([name, *[c for s in ("all", "narrow")
                          for c in (_num(100 * t.loc[(s, meth), "inside"], 0), _num(t.loc[(s, meth), "mad_vp"]))]])
            for meth, name in QUOTE_NAMES.items()]


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
        if (tables / f"exp_forecast_{a}.csv").exists():
            files.update(forecast=tab_forecast(tables, a), dyn=tab_dyn(tables, a), neural=tab_neural(tables, a))
        if (tables / f"exp_rmse_holdout_{a}.csv").exists():
            files.update(holdout=tab_holdout(tables, a), forecast_holdout=tab_forecast(tables, f"holdout_{a}"))
        if (tables / f"exp_quotes_{a}.csv").exists():
            files["quotes"] = tab_quotes(tables, a)
        for name, rows in files.items():
            (out / f"exp_tab_{name}_{a}.tex").write_text("\n".join(rows) + "\n")
