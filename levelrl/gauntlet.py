from __future__ import annotations

import json
import os
import random

import chess
import numpy as np
import torch

from levelrl import arena, maia, progress

# Maia against the Stockfish rungs, many games at once. Maia's moves are one batched forward over every game where
# it is to move, while the rungs' moves for the other games run in a process pool alongside it.


def sampler(cfg, model, policy: dict, gen: torch.Generator, bsz: int = int(os.environ.get("LEVELRL_MOVER_BSZ", "1024"))):
    # Mover for the arena: one batched forward over every game where this policy is to move, then a sample.
    # policy may carry a per-spec self_elo through spec["self_elo"].
    dev = cfg.device

    def mover(games):
        out = []
        for i in range(0, len(games), bsz):
            ch = games[i:i + bsz]
            x = torch.stack([g["hist"].tokens() for g in ch])
            m = torch.stack([g["hist"].legal_mask() for g in ch])
            se = [g["spec"].get("self_elo", policy["self_elo"]) for g in ch]
            oe = [g["spec"].get("oppo_elo", policy["oppo_elo"]) for g in ch]
            logits, _ = maia.run(model, x, m, se, oe, dev)
            temps = [g["spec"].get("temp", policy["temp"]) for g in ch]
            temp = temps[0] if len(set(temps)) == 1 else torch.tensor(temps, device=logits.device)[:, None]
            for g, j in zip(ch, sample(logits, temp, policy["top_p"], gen).tolist()):
                out.append(maia.to_move(g["board"], j))
        return out
    return mover


def sample(logits: torch.Tensor, temp: float, top_p: float, gen: torch.Generator) -> torch.Tensor:
    # Batched maia3.uci.sample_from_logits: nucleus after temperature, top move always kept.
    # temp is one number, or a column with one temperature a row.
    if not torch.is_tensor(temp) and temp <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits / temp, -1)
    if top_p < 1.0:
        sp, si = p.sort(-1, descending=True)
        keep = sp.cumsum(-1) <= top_p
        keep[:, 0] = True
        p = torch.zeros_like(p).scatter(1, si, sp * keep)
    return torch.multinomial(p, 1, generator=gen)[:, 0]


def play(cfg, model, policy: dict, opponents: list[str], openings: list[list[str]], seed: int = 0,
         name: str = "gauntlet", elo: dict[str, float] | None = None, label: str | None = None,
         extra: dict | None = None, cycle: bool = False) -> list[dict]:
    # Every opening is played with both colours. Each opening meets every opponent, or with cycle one opponent
    # in turn, which keeps a 200-opening suite at 400 games.
    rng = random.Random(seed)
    pairs = ([(opponents[i % len(opponents)], op) for i, op in enumerate(openings)] if cycle
             else [(o, op) for o in opponents for op in openings])
    todo = [{"opp": o, "opening": op, "white": w, "seed": rng.getrandbits(31), **(extra or {})}
            for o, op in pairs for w in (True, False)]
    return run(cfg, model, policy, todo, seed, name, elo, label or f"maia@{policy['self_elo']}", ", ".join(opponents))


def run(cfg, model, policy: dict, todo: list[dict], seed: int = 0, name: str = "gauntlet",
        elo: dict[str, float] | None = None, label: str = "maia", versus: str = "the ladder") -> list[dict]:
    # Plays a prepared list of game specs. A spec may carry its own self_elo, oppo_elo and label, so several dial
    # settings share one arena. A gauntlet on its own spends much of its time on its last few long games with the
    # engine workers idle, and sharing the arena overlaps those tails.
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    todo = list(todo)
    total, done, est = len(todo), [], None
    a = arena.Arena(cfg, sampler(cfg, model, policy, gen), lambda: todo.pop() if todo else None, name,
                    me=lambda s: s.get("label", label))
    progress.tick(name, 0, total, "games", force=True, note=f"{label} vs {versus} · starting")
    while not a.done():
        new = a.step()
        if not new:
            continue  # steps come many times a second now, the stats only move when a game ends
        done += new
        if elo and len(done) % 20 < len(new):
            est = rate(done, elo)
        stats = {"score": f"{sum(x['score'] for x in done) / len(done):.0%}",
                 "draws": f"{sum(x['score'] == 0.5 for x in done) / len(done):.0%}",
                 "longest": f"{max(x['plies'] for x in done)} plies"}
        if est is not None:
            stats = {"ladder elo so far": f"{est:.0f}", **stats}
        progress.tick(name, len(done), total, "games", stats=stats,
                      note=f"{label} vs {versus} · {len(a.live)} boards at once")
    return done


def rate(games: list[dict], elo: dict[str, float]) -> float:
    # Maximum-likelihood Elo of one player against rungs held at their calibrated ratings, draws as half points.
    # The score is monotone in the rating, so bisection finds it. Clamped so a clean sweep stays finite.
    def excess(r):
        return sum(g["score"] - 1 / (1 + 10 ** ((elo[g["opp"]] - r) / 400)) for g in games)
    lo, hi = -1000.0, 4500.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def rate_ci(games: list[dict], elo: dict[str, float], n: int = 1000, seed: int = 0) -> tuple[float, list[float]]:
    rng = random.Random(seed)
    samples = [rng.choices(games, k=len(games)) for _ in range(n)]
    opponents = np.asarray([[elo[g["opp"]] for g in sample] for sample in samples])
    scores = np.asarray([[g["score"] for g in sample] for sample in samples])
    lo, hi = np.full(n, -1000.0), np.full(n, 4500.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        excess = scores.sum(axis=1) - (1 / (1 + 10 ** ((opponents - mid[:, None]) / 400))).sum(axis=1)
        lo, hi = np.where(excess > 0, mid, lo), np.where(excess > 0, hi, mid)
    v = np.sort((lo + hi) / 2)
    return rate(games, elo), [float(v[int(0.025 * n)]), float(v[int(0.975 * n)])]


def save(path, policy, games, elo):
    r, ci = rate_ci(games, elo)
    out = {"policy": policy, "elo": r, "ci": ci, "games": len(games),
           "score": sum(g["score"] for g in games) / len(games),
           "by_opp": {o: {"n": sum(g["opp"] == o for g in games),
                          "score": sum(g["score"] for g in games if g["opp"] == o)
                          / max(1, sum(g["opp"] == o for g in games))} for o in sorted({g["opp"] for g in games})}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    path.with_suffix(".games.json").write_text(json.dumps(games))
    return out


def closest(elo: dict[str, float], target: float, k: int = 3, refused=()) -> list[str]:
    ok = [n for n in elo if n not in refused]
    return sorted(ok, key=lambda n: abs(elo[n] - target))[:k]

