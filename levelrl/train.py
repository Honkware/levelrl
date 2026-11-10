from __future__ import annotations

import copy
import json
from collections import deque
import math
import random
import time
from pathlib import Path

import torch

from levelrl import arena, evals, gauntlet, maia, progress

# GRPO on Maia-3-79M. Groups of G games start from one booked opening against one rung, half with each colour.
# The loss is w_out * (PPO on the outcome, pulled toward the band) + w_fid * KL_COEF * KL(policy || base at T).
#
# Outcome: the advantage is s * outcome, standardised inside the group, where s in [-1, 1] is the pull: positive
# while the policy plays below T, negative above it, zero on it. The spec's plain +outcome only ever pushes up,
# and the dial already plays above its setting at the low end. s comes from a rating of the last CTRL_GAMES
# training games, refreshed every update. The first run set it from a gauntlet every 25 updates and overshot the
# band by 100 to 250 Elo each way between evals (1720, 1470, 1602, 1480 for one arm).
#
# Fidelity: the first run rewarded the base model's log-prob of the sampled move (the judge). That is only the
# cross-entropy half of a KL, so the policy raised it by sharpening: entropy fell (1.28 to 0.90 on the judge-only
# arm), strength rose with it, and move-match on real humans dropped 0.4 to 2.4 points on every arm. The whole
# KL has its optimum at the base model itself, and it is exact per position, so it is a loss term here and not a
# sampled reward. The judge's score is still logged.
# Outcome credit decays by GAMMA a ply back from the end and has the WDL head's prediction subtracted with a
# coefficient fitted on the previous batch, kept in [0, 1].
# Profiling: the 4090 runs the model at ~4k positions/s and the step is bound by Stockfish and Python, so
# torch.compile buys nothing measurable here and is left out. Incremental encoding is on (see maia.Hist).

G = 8
MIN_BATCH = 32
LR = 1e-5
CLIP = 0.2
CLIP_HIGH_DAPO = 0.28
EPOCHS = 2
MB = 64
EMA = 0.95
GAMMA = 0.99
CKPT_EVERY = 25
EVAL_AT = (25, 50, 75)
EVAL_OPENINGS = 34
RESIGN_P, RESIGN_N = 0.97, 6
PULL = 150.0  # Elo from the band at which the outcome term is at full strength
KL_COEF = 4.0
CTRL_GAMES, CTRL_MIN = 256, 64  # training games behind the running rating, and how many before it is trusted
ENTROPY_FLOOR = 0.3  # collapse if entropy falls below this fraction of its first value


def _z(v: list[float]) -> list[float]:
    if not v:
        return []
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    return [0.0] * len(v) if sd < 1e-6 else [(x - m) / sd for x in v]


def pull(T: float, elo: float) -> float:
    return max(-1.0, min(1.0, (T - elo) / PULL))


class Trainer:
    def __init__(self, cfg, base, out: Path, T: int, w_out: float, w_fid: float, sign: float, units, games: int,
                 seed: int = 0, name: str = "train"):
        self.cfg, self.base, self.out, self.T, self.w_out, self.w_fid = cfg, base, out, T, w_out, w_fid
        self.units, self.games, self.seed, self.name = units, games, seed, name
        self.dev = cfg.device
        self.policy = copy.deepcopy(base).float()
        for p in self.policy.parameters():
            p.requires_grad_(True)  # the base it is copied from is frozen
        self.ema = copy.deepcopy(base).float().eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=LR, fused=str(self.dev).startswith("cuda"))
        self.st = {"update": 0, "games": 0, "sign": sign, "c": 0.0, "lr": LR, "clip_high": CLIP, "recoveries": 0,
                   "entropy0": None, "log": [], "evals": []}
        self.rng = random.Random(seed)
        self.recent = deque(maxlen=CTRL_GAMES)
        self.elo = {n: e for e, n in units.rungs}
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.out.mkdir(parents=True, exist_ok=True)
        self.resume()

    # --- checkpoints ---
    def save(self, tag: str | None = None):
        d = {"policy": self.policy.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
             "st": self.st, "rng": self.rng.getstate()}
        tmp = self.out / "last.pt.tmp"
        torch.save(d, tmp)
        tmp.replace(self.out / "last.pt")
        if tag:
            torch.save(self.ema.state_dict(), self.out / f"ema_{tag}.pt")

    def resume(self):
        p = self.out / "last.pt"
        if p.exists():
            d = torch.load(p, map_location=self.dev, weights_only=False)
            self.policy.load_state_dict(d["policy"])
            self.ema.load_state_dict(d["ema"])
            self.opt.load_state_dict(d["opt"])
            self.st = d["st"]
            self.rng.setstate(d["rng"])
            for g in self.opt.param_groups:
                g["lr"] = self.st["lr"]

    # --- rollouts ---
    def opponent(self) -> str:
        # Curriculum: the opponent's Lichess strength ramps from T-400 to T over the first third of training.
        f = min(1.0, self.st["games"] / max(1, self.games / 3))
        return self.units.rung_near(self.T - 400 * (1 - f), self.rng)

    def feed_fn(self, openings):
        q = []
        gid = [0]

        def feed():
            if not q:
                op = self.rng.choice(openings)
                opp = self.opponent()
                gid[0] += 1
                for k in range(G):
                    q.append({"opp": opp, "opening": op, "white": k % 2 == 0, "seed": self.rng.getrandbits(31),
                              "group": f"{self.st['update']}-{gid[0]}"})
            return q.pop()
        return feed

    @torch.no_grad()
    def mover(self, games):
        out = []
        for i in range(0, len(games), 512):
            ch = games[i:i + 512]
            x = torch.stack([g["hist"].tokens() for g in ch])
            m = torch.stack([g["hist"].legal_mask() for g in ch])
            n = len(ch)
            self.policy.eval()
            logits, value = maia.run(self.policy, x, m, [self.T] * n, [self.T] * n, self.dev)
            blog, _ = maia.run(self.base, x, m, [self.T] * n, [self.T] * n, self.dev)
            idx = gauntlet.sample(logits, 1.0, 1.0, self.gen)
            lp = torch.log_softmax(logits, -1).gather(1, idx[:, None])[:, 0]
            jl = torch.log_softmax(blog, -1).gather(1, idx[:, None])[:, 0]
            pv = torch.softmax(value, -1)  # [loss, draw, win] for the side to move
            v = (pv[:, 2] - pv[:, 0]).tolist()
            ploss = pv[:, 0].tolist()
            for k, g in enumerate(ch):
                g["lowc"] = g.get("lowc", 0) + 1 if ploss[k] > RESIGN_P else 0
                if g["lowc"] >= RESIGN_N:
                    g["resign"] = True
                    out.append(None)
                    continue
                g["rec"].append((x[k].to(torch.bfloat16), m[k], int(idx[k]), float(lp[k]), v[k], float(jl[k]),
                                 g["board"].ply()))
                out.append(maia.to_move(g["board"], int(idx[k])))
        return out

    # --- update ---
    def advantages(self, group: list[dict]):
        s, c = self.st["sign"], self.st["c"]
        d = 1.0 if s >= 0 else -1.0
        per, rs = [], []
        for g in group:
            o = 2 * g["score"] - 1
            rs.append(s * o)
            for r in g["rec"]:
                per.append((r, d * (o * GAMMA ** (g["plies"] - r[6]) - c * r[4])))
        zo = _z([p[1] for p in per])
        return [(p[0], self.w_out * abs(s) * x) for p, x in zip(per, zo)], rs

    def fit_c(self, batch: list[dict]):
        ys, vs = [], []
        for g in batch:
            o = 2 * g["score"] - 1
            for r in g["rec"]:
                ys.append(o * GAMMA ** (g["plies"] - r[6]))
                vs.append(r[4])
        if len(vs) > 10:
            mv, my = sum(vs) / len(vs), sum(ys) / len(ys)
            var = sum((v - mv) ** 2 for v in vs)
            cov = sum((v - mv) * (y - my) for v, y in zip(vs, ys))
            self.st["c"] = min(1.0, max(0.0, cov / var)) if var > 0 else 0.0

    def update(self, groups: list[list[dict]]) -> dict:
        items, rewards = [], []
        for grp in groups:
            a, rs = self.advantages(grp)
            items += a
            rewards += rs
        self.fit_c([g for grp in groups for g in grp])  # applies from the next batch, so this one stays held out
        self.policy.eval()  # same mode the old log-probs were taken in, so the ratio starts at exactly 1
        stats = {"loss": 0.0, "entropy": 0.0, "kl_base": 0.0, "clipfrac": 0.0, "n": 0}
        for _ in range(EPOCHS):
            self.rng.shuffle(items)
            for i in range(0, len(items), MB):
                mb = items[i:i + MB]
                x = torch.stack([r[0] for r, _ in mb]).float()
                m = torch.stack([r[1] for r, _ in mb])
                idx = torch.tensor([r[2] for r, _ in mb], device=self.dev)
                old = torch.tensor([r[3] for r, _ in mb], device=self.dev)
                adv = torch.tensor([a for _, a in mb], device=self.dev)
                n = len(mb)
                logits, _ = maia.run(self.policy, x, m, [self.T] * n, [self.T] * n, self.dev)
                logp_all = torch.log_softmax(logits, -1)
                lp = logp_all.gather(1, idx[:, None])[:, 0]
                ratio = torch.exp(lp - old)
                clipped = torch.clamp(ratio, 1 - CLIP, 1 + self.st["clip_high"])
                with torch.no_grad():
                    blog, _ = maia.run(self.base, x, m, [self.T] * n, [self.T] * n, self.dev)
                    bl = torch.log_softmax(blog, -1).clamp_min(-1e4)
                # Illegal moves are -inf on both sides. Clamped, they cancel to zero and carry no gradient.
                kl = (logp_all.exp() * (logp_all.clamp_min(-1e4) - bl)).sum(-1).mean()
                loss = -torch.min(ratio * adv, clipped * adv).mean() + self.w_fid * KL_COEF * kl
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    ema_params = list(self.ema.parameters())
                    torch._foreach_mul_(ema_params, EMA)
                    torch._foreach_add_(ema_params, list(self.policy.parameters()), alpha=1 - EMA)
                    p = logp_all.exp()
                    ent = -(p * logp_all.nan_to_num(0.0, neginf=0.0)).sum(-1).mean()
                    stats["loss"] += float(loss) * n
                    stats["entropy"] += float(ent) * n
                    stats["kl_base"] += float(kl) * n
                    stats["clipfrac"] += float(((ratio - 1).abs() > CLIP).float().mean()) * n
                    stats["n"] += n
        n = max(1, stats.pop("n"))
        out = {k: v / n for k, v in stats.items()}
        out["reward"] = sum(rewards) / max(1, len(rewards))
        out["judge"] = sum(r[5] for grp in groups for g in grp for r in g["rec"]) / max(1, len(items))
        out["positions"] = len(items)
        return out

    # --- the loop ---
    def healthy(self, s: dict) -> bool:
        if not all(math.isfinite(v) for v in s.values()):
            return False
        if self.st["entropy0"] is None:
            self.st["entropy0"] = s["entropy"]
        return s["entropy"] >= ENTROPY_FLOOR * self.st["entropy0"]

    def recover(self, why: str):
        # Entropy collapse or a NaN: back to the last checkpoint at half the learning rate, and the second time
        # also switch on DAPO's asymmetric clip-higher.
        rec = self.st["recoveries"] + 1
        self.resume()
        self.st["recoveries"] = rec
        self.st["lr"] /= 2
        for g in self.opt.param_groups:
            g["lr"] = self.st["lr"]
        if rec >= 2:
            self.st["clip_high"] = CLIP_HIGH_DAPO
        self.st["log"].append({"update": self.st["update"], "recovery": why, "lr": self.st["lr"],
                               "clip_high": self.st["clip_high"]})
        progress.event(self.name, f"{why}: back to update {self.st['update']}, lr {self.st['lr']:.1e}")

    def steer(self):
        # The pull follows a rating of the most recent training games, so it reacts within a few updates.
        if len(self.recent) >= CTRL_MIN:
            est = self.units.lich(gauntlet.rate(list(self.recent), self.elo))
            self.st["est"], self.st["sign"] = est, pull(self.T, est)

    def evaluate(self, tag: str, on_eval):
        if any(e.get("tag") == tag for e in self.st["evals"]):
            return
        self.save(tag=tag)
        self.st["evals"].append(on_eval(self, tag))
        self.save()

    def report(self, note: str, stats: dict | None = None, force: bool = False):
        s = {"T": self.T, "w_out/w_fid": f"{self.w_out}/{self.w_fid}", "toward band": f"{self.st['sign']:+.2f}"}
        if stats:
            s.update(stats)
        progress.tick(self.name, min(self.st["games"], self.games), self.games, "games", note=note, stats=s, force=force)

    def run(self, openings, on_eval) -> dict:
        a = arena.Arena(self.cfg, self.mover, self.feed_fn(openings), self.name, me=lambda s: f"policy@{self.T}")
        groups: dict[str, list] = {}
        ready = []
        t0 = time.time()
        self.report(f"update {self.st['update']} · collecting rollouts", force=True)
        if not (self.out / "last.pt").exists():
            self.save()  # recover() needs something to go back to before the first periodic checkpoint
        if self.st["update"] in EVAL_AT:
            self.evaluate(str(self.st["update"]), on_eval)  # a run that died inside this eval resumes right here
        while self.st["games"] < self.games:
            for g in a.step():
                groups.setdefault(g["group"], []).append(g)
                if len(groups[g["group"]]) == G:
                    grp = groups.pop(g["group"])
                    self.recent.extend({"opp": x["opp"], "score": x["score"]} for x in grp)
                    # Dynamic sampling: with no fidelity term, a group where every game ended the same way has
                    # nothing to learn from. With one, its positions still feed the KL.
                    if self.w_fid > 0 or len({x["score"] for x in grp}) > 1:
                        ready.append(grp)
                    self.st["games"] += G
            self.report(f"update {self.st['update']} · collecting rollouts · {len(ready) * G}/{MIN_BATCH} update games")
            if len(ready) * G < MIN_BATCH:
                continue
            self.steer()
            s = self.update(ready)
            ready = []
            if not self.healthy(s):
                self.recover("NaN" if not all(math.isfinite(v) for v in s.values()) else "entropy collapse")
                continue
            self.st["update"] += 1
            u = self.st["update"]
            s.update(update=u, games=self.st["games"], c=self.st["c"], sign=self.st["sign"], t=time.time() - t0)
            if self.st.get("est") is not None:  # absent until the running rating has enough games
                s["est"] = self.st["est"]
            self.st["log"].append(s)
            self.report(f"update {u} · reward {s['reward']:+.3f} · KL {s['kl_base']:.4f} · entropy {s['entropy']:.2f}",
                        stats={"judge": f"{s['judge']:.2f}", "cv c": f"{self.st['c']:.2f}"})
            if u % CKPT_EVERY == 0:
                self.save()
            if u in EVAL_AT:
                self.evaluate(str(u), on_eval)
        self.evaluate("final", on_eval)
        (self.out / "train_log.json").write_text(json.dumps({k: v for k, v in self.st.items()}, indent=1))
        return self.st
