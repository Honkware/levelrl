from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import torch

from levelrl import arena, evals, gauntlet, maia, progress

# GRPO on Maia-3-79M. Groups of G games start from one booked opening against one rung, half with each colour.
# Per game r = w_out * s * outcome + w_fid * fidelity, where fidelity is the frozen base model's mean log-prob, at
# the target band, of the moves the policy played (the judge), and s points the outcome toward the band: +1 while
# the last measured Lichess Elo is below T, -1 above it. The spec's plain +outcome only ever pushes up, and the
# dial already plays above its setting at the low end, so for T=1100 it would push away from the band.
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
ENTROPY_FLOOR = 0.3  # collapse if entropy falls below this fraction of its first value


class Trainer:
    def __init__(self, cfg, base, out: Path, T: int, w_out: float, w_fid: float, sign: int, units, games: int,
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
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=LR)
        self.st = {"update": 0, "games": 0, "sign": sign, "c": 0.0, "lr": LR, "clip_high": CLIP, "recoveries": 0,
                   "entropy0": None, "log": [], "evals": []}
        self.rng = random.Random(seed)
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
            m = torch.stack([maia.legal(g["board"]) for g in ch])
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
        per, rs = [], []
        for g in group:
            o = 2 * g["score"] - 1
            fid = sum(r[5] for r in g["rec"]) / max(1, len(g["rec"]))
            rs.append(self.w_out * s * o + self.w_fid * fid)
            for r in g["rec"]:
                k = g["plies"] - r[6]
                per.append((r, s * (o * GAMMA ** k - c * r[4]), fid))
        if not per:
            return [], rs
        mo = sum(p[1] for p in per) / len(per)
        mf = sum(p[2] for p in per) / len(per)
        a = [self.w_out * (u - mo) + self.w_fid * (f - mf) for _, u, f in per]
        sd = math.sqrt(sum(x * x for x in a) / len(a)) + 1e-6
        return [(p[0], x / sd) for p, x in zip(per, a)], rs

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
                loss = -torch.min(ratio * adv, clipped * adv).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    for pe, pp in zip(self.ema.parameters(), self.policy.parameters()):
                        pe.mul_(EMA).add_(pp.detach(), alpha=1 - EMA)
                    p = logp_all.exp()
                    ent = -(p * logp_all.nan_to_num(0.0, neginf=0.0)).sum(-1).mean()
                    blog, _ = maia.run(self.base, x, m, [self.T] * n, [self.T] * n, self.dev)
                    bl = torch.log_softmax(blog, -1)
                    kl = (p * (logp_all - bl).nan_to_num(nan=0.0, neginf=0.0, posinf=0.0)).sum(-1).mean()
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

    def run(self, openings, on_eval) -> dict:
        a = arena.Arena(self.cfg, self.mover, self.feed_fn(openings), self.name, me=lambda s: f"policy@{self.T}")
        groups: dict[str, list] = {}
        ready = []
        t0 = time.time()
        while self.st["games"] < self.games:
            for g in a.step():
                groups.setdefault(g["group"], []).append(g)
                if len(groups[g["group"]]) == G:
                    grp = groups.pop(g["group"])
                    o = [self.w_out * self.st["sign"] * (2 * x["score"] - 1)
                         + self.w_fid * sum(r[5] for r in x["rec"]) / max(1, len(x["rec"])) for x in grp]
                    if max(o) - min(o) > 1e-9:  # dynamic sampling: a group with no reward spread teaches nothing
                        ready.append(grp)
                    self.st["games"] += G
            if len(ready) * G < MIN_BATCH:
                continue
            s = self.update(ready)
            ready = []
            if not self.healthy(s):
                self.recover("NaN" if not all(math.isfinite(v) for v in s.values()) else "entropy collapse")
                continue
            self.st["update"] += 1
            u = self.st["update"]
            s.update(update=u, games=self.st["games"], c=self.st["c"], sign=self.st["sign"], t=time.time() - t0)
            self.st["log"].append(s)
            progress.tick(self.name, min(self.st["games"], self.games), self.games, "games",
                          note=f"update {u} · reward {s['reward']:+.3f} · KL {s['kl_base']:.4f} · entropy {s['entropy']:.2f}",
                          stats={"T": self.T, "w_out/w_fid": f"{self.w_out}/{self.w_fid}", "toward band": f"{self.st['sign']:+d}",
                                 "judge": f"{s['judge']:.2f}", "cv c": f"{self.st['c']:.2f}"})
            if u % CKPT_EVERY == 0:
                self.save()
            if u in EVAL_AT:
                self.save(tag=str(u))
                self.st["evals"].append(on_eval(self, str(u)))
                self.save()
        self.save(tag="final")
        self.st["evals"].append(on_eval(self, "final"))
        self.save()
        (self.out / "train_log.json").write_text(json.dumps({k: v for k, v in self.st.items()}, indent=1))
        return self.st
