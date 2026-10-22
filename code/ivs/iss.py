"""Минимальный клиент MOEX ISS (https://iss.moex.com) с дисковым кэшем.

Исторические ответы ISS неизменны, поэтому кэшируются целиком (json.gz);
текущие данные (marketdata) запрашиваются с ``cache=False``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

BASE_URL = "https://iss.moex.com/iss"


class SubscriptionRequired(RuntimeError):
    """ISS вернул HTML-страницу «Информация доступна только подписчикам»."""


class ISSClient:
    def __init__(self, cache_dir: str | Path | None = None, retries: int = 6, timeout: float = 60.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.timeout = timeout
        self._local = threading.local()

    @property
    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json.gz"

    def get_json(self, path: str, params: dict | None = None, cache: bool = True) -> dict:
        params = {"iss.meta": "off", **(params or {})}
        url = f"{BASE_URL}/{path}.json?{urlencode(sorted(params.items()))}"
        cache_file = self._cache_path(url) if cache else None
        if cache_file is not None and cache_file.exists():
            with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                return json.load(f)

        data = self._fetch(url)
        if cache_file is not None:
            cache_file.parent.mkdir(exist_ok=True)
            tmp = cache_file.with_name(cache_file.name + f".{threading.get_ident()}.tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            tmp.replace(cache_file)
        return data

    def _fetch(self, url: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self._session.get(url, timeout=self.timeout)
                if r.status_code == 200 and "html" in r.headers.get("content-type", ""):
                    raise SubscriptionRequired(url)
                r.raise_for_status()
                return json.loads(r.content.decode("utf-8-sig"))
            except SubscriptionRequired:
                raise
            except (requests.RequestException, json.JSONDecodeError) as e:
                last_error = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"ISS request failed: {url}") from last_error

    def table(self, path: str, block: str, params: dict | None = None, cache: bool = True) -> pd.DataFrame:
        """Таблица ``block`` со всеми страницами (постранично через ``start``)."""
        params = dict(params or {})
        frames, start = [], 0
        while True:
            data = self.get_json(path, {**params, "start": start}, cache=cache)
            t = data[block]
            if t["data"] or not frames:
                frames.append(pd.DataFrame(t["data"], columns=t["columns"]))
            cursor = data.get(f"{block}.cursor")
            if cursor and cursor["data"]:
                index, total, page_size = cursor["data"][0]
                start = index + page_size
                if start >= total:
                    break
            elif len(t["data"]) == 0 or len(t["data"]) < 100:
                break
            else:
                start += len(t["data"])
        return pd.concat(frames, ignore_index=True)

    # --- срочный рынок -------------------------------------------------------

    def options_history(self, date: str, assetcode: str) -> pd.DataFrame:
        """Итоги дня по всем опционам на базовый актив (без bid/ask: их ISS в истории не отдаёт)."""
        return self.table(
            "history/engines/futures/markets/options/securities",
            "history",
            {"date": date, "assetcode": assetcode},
        )

    def futures_history(self, date: str, assetcode: str) -> pd.DataFrame:
        return self.table(
            "history/engines/futures/markets/forts/securities",
            "history",
            {"date": date, "assetcode": assetcode},
        )

    def security_description(self, secid: str) -> dict:
        data = self.get_json(f"securities/{secid}", {"iss.only": "description"})
        return {row[0]: row[2] for row in data["description"]["data"]}
