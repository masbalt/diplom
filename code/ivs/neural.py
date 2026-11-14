"""Нейросетевой метод сравнения: сеть с вниманием по множеству видимых сделок.

Задача и входы те же, что у предлагаемого метода (ivs.gp): по видимым сделкам дня и начальному
приближению восстановить IV в произвольных точках. Вместо гауссовского процесса с заданным видом
ядра — сеть, которая сама учится, как взвешивать видимые сделки (нейронный процесс с вниманием:
Garnelo et al., 2018; Kim et al., 2019).

Вход: видимая сделка — (x, ln T, остаток r = IV − начальное приближение, число минут со сделками);
точка запроса — (x, ln T). В варианте без начального приближения остаток считается от средней IV
видимых сделок. Пара «запрос, сделка» описывается разностью x, разностью ln T и признаком «та же серия».

Сеть: два слоя внимания между видимыми сделками, два слоя внимания от точек запроса к сделкам,
затем выход вида  gate(q) * sum_i a_qi r_i + free(q),  где веса a_qi неотрицательны и дают в сумме 1.
Первое слагаемое — взвешенное среднее остатков видимых сделок, как у ядерного сглаживания.

Обучение: дни обучающего периода. На каждом шаге часть сделок дня скрывается по тем же схемам,
что в эксперименте (случайная доля, крылья, целая серия, дальние серии, самые торгуемые), сеть
восстанавливает скрытые сделки по видимым; функция потерь — Хьюбера. Итоговый прогноз — среднее
по нескольким сетям, обученным с разными начальными значениями весов.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch import nn

SCALE = 100.0        # остатки подаются в сеть в п.п. волатильности
PAIR_FEATURES = 6
_nets: dict = {}     # сети, уже собранные в этом процессе (ключ — имя варианта)


def tensors(day: pd.DataFrame, prior: str | None) -> dict:
    """Массивы одного набора точек. ``order`` — место контракта в списке самых торгуемых."""
    f32 = lambda col: torch.tensor(day[col].to_numpy(float), dtype=torch.float32)   # noqa: E731
    out = {"x": f32("x"), "lnT": torch.log(f32("T")),
           "e": torch.tensor(day["expiry"].to_numpy().astype("datetime64[D]").astype(np.int64)),
           "prior": f32(prior) if prior else torch.zeros(len(day))}
    for col, name in (("iv", "iv"), ("n_candles", "n"), ("z", "z")):
        if col in day:
            out[name] = f32(col)
    if "vol_sync" in day:
        order = np.lexsort((-day["vol_sync"].to_numpy(), -day["n_candles"].to_numpy()))
        out["order"] = torch.tensor(np.argsort(order))
    return out


def _take(t: dict, mask: torch.Tensor) -> dict:
    return {k: v[mask] for k, v in t.items()}


def _residuals(vis: dict, use_prior: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Остатки видимых сделок в п.п. и уровень, от которого они отсчитаны (в варианте без
    начального приближения — средняя IV видимых сделок)."""
    base = torch.zeros(()) if use_prior else vis["iv"].mean()
    return SCALE * (vis["iv"] - vis["prior"] - base), base


def _pairs(a: dict, b: dict) -> torch.Tensor:
    dx = a["x"][:, None] - b["x"][None, :]
    dT = (a["lnT"][:, None] - b["lnT"][None, :]).abs()
    same = (a["e"][:, None] == b["e"][None, :]).float()
    return torch.stack([3 * dx, 3 * dx.abs(), same, dT, torch.exp(-3 * dT), 9 * a["x"][:, None] * b["x"][None, :]], -1)


class Attention(nn.Module):
    """Слой внимания от точек ``hq`` к сделкам ``hc``; признаки пары добавляются к весам внимания."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))
        self.bias = nn.Sequential(nn.Linear(PAIR_FEATURES, 32), nn.GELU(), nn.Linear(32, heads))
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm_q, self.norm_c, self.norm_ff = (nn.LayerNorm(dim) for _ in range(3))

    def forward(self, hq, hc, pairs):
        split = lambda t: t.view(t.shape[0], self.heads, -1).transpose(0, 1)   # noqa: E731
        q, c = split(self.q(self.norm_q(hq))), self.norm_c(hc)
        k, v = split(self.k(c)), split(self.v(c))
        a = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]) + self.bias(pairs).permute(2, 0, 1)
        hq = hq + self.out((a.softmax(-1) @ v).transpose(0, 1).reshape(hq.shape[0], -1))
        return hq + self.ff(self.norm_ff(hq))


class Net(nn.Module):
    def __init__(self, dim: int = 48, heads: int = 4, layers: int = 2):
        super().__init__()
        self.enc_c = nn.Sequential(nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim))
        self.enc_q = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.among = nn.ModuleList([Attention(dim, heads) for _ in range(layers)])
        self.cross = nn.ModuleList([Attention(dim, heads) for _ in range(layers)])
        self.key_q, self.key_c = nn.Linear(dim, 32), nn.Linear(dim, 32)
        self.score = nn.Sequential(nn.Linear(PAIR_FEATURES, 32), nn.GELU(), nn.Linear(32, 1))
        self.gate = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.free = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, vis: dict, r: torch.Tensor, qry: dict) -> torch.Tensor:
        """Поправка в точках ``qry`` (п.п.) по видимым сделкам ``vis`` с остатками ``r``."""
        xc, xq = 3 * vis["x"], 3 * qry["x"]
        hc = self.enc_c(torch.stack([xc, xc**2, vis["lnT"] / 2 + 1, r, r.abs(), vis["n"].rsqrt(),
                                     torch.log1p(vis["n"]) / 3], -1))
        hq = self.enc_q(torch.stack([xq, xq**2, qry["lnT"] / 2 + 1], -1))
        pcc, pqc = _pairs(vis, vis), _pairs(qry, vis)
        for layer in self.among:
            hc = layer(hc, hc, pcc)
        for layer in self.cross:
            hq = layer(hq, hc, pqc)
        a = self.key_q(hq) @ self.key_c(hc).T / math.sqrt(32) + self.score(pqc).squeeze(-1)
        return torch.sigmoid(self.gate(hq).squeeze(-1) + 1.0) * (a.softmax(-1) @ r) + self.free(hq).squeeze(-1)


# --- обучение ---------------------------------------------------------------------------

def sample_mask(t: dict, rng: np.random.Generator, p: dict) -> torch.Tensor:
    """Случайный набор скрытых сделок дня: те же схемы, что в эксперименте (ivs.experiment.mask_configs)."""
    n, u = len(t["x"]), rng.random()
    hid = np.zeros(n, bool)
    if u < 0.6:
        hid[rng.choice(n, int(np.clip(round(rng.uniform(0.05, 0.95) * n), 1, n - 1)), replace=False)] = True
    elif u < 0.7:
        hid = (t["z"].abs() > p["WINGS_Z"]).numpy()
    elif u < 0.8:
        hid = (t["e"] == int(rng.choice(np.unique(t["e"].numpy())))).numpy()
    elif u < 0.9:
        hid = ~np.isin(t["lnT"].numpy(), np.sort(np.unique(t["lnT"].numpy()))[: p["FAR_KEEP"]])
    else:
        hid = (t["order"] >= int(rng.choice(p["TOPN"]))).numpy()
    if hid.all() or not hid.any():   # схема неприменима к этому дню — скрывается случайная половина
        hid = np.zeros(n, bool)
        hid[rng.choice(n, max(1, n // 2), replace=False)] = True
    return torch.tensor(hid)


def fit(job: tuple) -> dict:
    """Обучает одну сеть. ``job`` = (дни обучения, колонка начального приближения или None, параметры, seed)."""
    days, prior, p, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [tensors(g, prior) for g in days]
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=p["NP_LR"], weight_decay=1e-4)
    steps = p["NP_EPOCHS"] * math.ceil(len(data) / p["NP_BATCH"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["NP_LR"], total_steps=steps)
    for _ in range(p["NP_EPOCHS"]):
        order = rng.permutation(len(data))
        for b in range(0, len(order), p["NP_BATCH"]):
            batch = order[b:b + p["NP_BATCH"]]
            loss = 0.0
            for i in batch:
                hid = sample_mask(data[i], rng, p)
                vis, qry = _take(data[i], ~hid), _take(data[i], hid)
                r, base = _residuals(vis, prior is not None)
                target = SCALE * (qry["iv"] - qry["prior"] - base)
                loss = loss + nn.functional.huber_loss(net(vis, r, qry), target, delta=p["NP_HUBER"])
            opt.zero_grad()
            (loss / len(batch)).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    return {k: v.numpy() for k, v in net.state_dict().items()}


def fit_all(train: pd.DataFrame, priors: dict[str, str | None], p: dict, workers: int) -> dict:
    """Ансамбли сетей для каждого варианта: {имя: {"prior": колонка, "states": [веса сетей]}}."""
    days = [g for _, g in train.groupby("date")]
    jobs = [(days, prior, p, seed) for prior in priors.values() for seed in range(p["NP_SEEDS"])]
    with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
        states = list(pool.map(fit, jobs))
    n = p["NP_SEEDS"]
    return {name: {"prior": prior, "states": states[i * n:(i + 1) * n]}
            for i, (name, prior) in enumerate(priors.items())}


# --- восстановление -----------------------------------------------------------------------

def _load(name: str, states: list[dict]) -> list[Net]:
    if name not in _nets:
        torch.set_num_threads(1)
        nets = []
        for state in states:
            net = Net()
            net.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
            nets.append(net.eval())
        _nets[name] = nets
    return _nets[name]


def predict(vis: pd.DataFrame, hid: pd.DataFrame, model: dict, name: str) -> np.ndarray:
    """IV в точках ``hid`` по видимым сделкам ``vis``: среднее по сетям ансамбля ``model``."""
    prior = model["prior"]
    v, q = tensors(vis, prior), tensors(hid, prior)
    with torch.no_grad():
        r, base = _residuals(v, prior is not None)
        corr = torch.stack([net(v, r, q) for net in _load(name, model["states"])]).mean(0) / SCALE
    return (q["prior"] + base + corr).numpy().astype(float)
