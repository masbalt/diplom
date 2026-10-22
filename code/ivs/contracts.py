"""Разбор кодов опционов MOEX и справочник серий (дата экспирации, базовый фьючерс).

Краткий код опциона на фьючерс: ``<БА><страйк><тип расчётов><месяц><год>[неделя][_ГГГГ]``,
например ``Si80000BL6``, ``RI107500BO0D``, ``Si61250BD6_2016``:

* месяц: A–L — колл (январь–декабрь), M–X — пут (январь–декабрь);
* год — последняя цифра; для старых контрактов ISS добавляет суффикс ``_ГГГГ``;
* неделя — буква недельной серии (A, B, D, E) или пусто для месячной/квартальной.

В истории ISS нет даты экспирации, поэтому для каждой группы (одна экспирация)
берётся описание одного представителя через ``/securities/{secid}``.
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .iss import ISSClient

SECID_RE = re.compile(
    r"^(?P<code>[A-Za-z]{2})(?P<strike>-?\d+(?:\.\d+)?)(?P<style>[A-Z])"
    r"(?P<month>[A-X])(?P<ydigit>\d)(?P<week>[A-Z]?)(?:_(?P<year>\d{4}))?$"
)
SHORTNAME_RE = re.compile(r"^(?P<underlying>.+?)M(?P<dd>\d{2})(?P<mm>\d{2})(?P<yy>\d{2})(?P<cp>[CP])(?P<ea>[AE])")


def parse_secids(secids: pd.Series) -> pd.DataFrame:
    """Страйк, тип опциона и ключ серии по коду. Нераспознанные коды дают NaN."""
    parts = secids.str.extract(SECID_RE)
    month_idx = parts["month"].map(lambda m: ord(m) - ord("A") if isinstance(m, str) else np.nan)
    out = pd.DataFrame(index=secids.index)
    out["code"] = parts["code"]
    out["strike"] = pd.to_numeric(parts["strike"], errors="coerce")
    out["opt_type"] = np.where(month_idx < 12, "C", "P")
    out.loc[month_idx.isna(), "opt_type"] = None
    out["month"] = month_idx % 12 + 1
    # Ключ экспирации общий для колла и пута: месяц приводится к номеру 1..12.
    out["group"] = (
        parts["code"] + "|" + parts["style"] + "|" + out["month"].astype("Int64").astype(str) + "|"
        + parts["ydigit"] + "|" + parts["week"].fillna("") + "|" + parts["year"].fillna("")
    )
    out.loc[parts["code"].isna(), "group"] = None
    return out


class SeriesRegistry:
    """Кэш соответствия «группа кода → экспирация, базовый фьючерс» в CSV."""

    COLUMNS = ["group", "secid", "expiry", "underlying", "shortname"]

    def __init__(self, path: str | Path, client: ISSClient):
        self.path = Path(path)
        self.client = client
        self._lock = threading.Lock()
        if self.path.exists():
            self.table = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        else:
            self.table = pd.DataFrame(columns=self.COLUMNS)

    def _describe(self, group: str, secid: str) -> dict:
        desc = self.client.security_description(secid)
        shortname = desc.get("SHORTNAME", "") or ""
        m = SHORTNAME_RE.match(shortname.strip())
        expiry = desc.get("LSTTRADE") or ""
        if not expiry and m:
            expiry = f"20{m['yy']}-{m['mm']}-{m['dd']}"
        return {
            "group": group,
            "secid": secid,
            "expiry": expiry,
            "underlying": m["underlying"] if m else "",
            "shortname": shortname,
        }

    def resolve(self, examples: dict[str, str], workers: int = 8) -> None:
        """Догружает описания для групп, которых ещё нет в справочнике."""
        known = set(self.table["group"])
        todo = [(g, s) for g, s in examples.items() if g not in known]
        if not todo:
            return
        with ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(lambda gs: self._describe(*gs), todo))
        with self._lock:
            self.table = pd.concat([self.table, pd.DataFrame(rows)], ignore_index=True)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.table.to_csv(self.path, index=False)

    def lookup(self) -> pd.DataFrame:
        t = self.table.copy()
        t["expiry"] = pd.to_datetime(t["expiry"], errors="coerce")
        return t.set_index("group")[["expiry", "underlying"]]
