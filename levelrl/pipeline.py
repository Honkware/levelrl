from __future__ import annotations

import concurrent.futures
import json
import os
import random
import shutil
import threading
import time
import traceback
import zlib
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

from levelrl import anchors, arena, data, evals, gauntlet, ladder, maia, openings, progress, units
from levelrl.train import BAND, CTRL_GAMES, ETA, KP, PULL, Trainer

# The whole run, unattended. Each stage writes runs/pipeline/<stage>.json when it finishes, so a rerun picks up
# where the last one stopped, and a sweep arm resumes from its last checkpoint. Nothing here publishes anything:
# the last stage writes results/ and a summary, and uploading waits for a person.

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "pipeline"
RES = ROOT / "results"
CALIB = ROOT / "runs" / "elo_calibration.json"
BOT = {"history": True, "temp": 1.1, "top_p": 0.94}  # data/anchors/bot_sampling.json
POLICY = {"history": True, "temp": 1.0, "top_p": 1.0}
BASE_DIALS = (1100, 1300, 1500, 1700, 1900)
GOALS = (0.25, 0.5, 0.75, 1.0)  # each arm's goal, as a share of the way from where the base model plays to T
ARM_GAMES = 4800
SWEEP_WORKERS = 2
SMOKE_GAMES = 320
EVAL_OPENINGS = 200
SENS_OPENINGS = 100
HUMANLIKE_MOVES = 16_000
STAGES = ("snapshot", "anchors", "crosscheck", "base", "baselines", "smoke", "sweep", "report")
OPTIONAL = ("crosscheck",)


def done(name):
    return (RUN / f"{name}.json").exists()


def save(name, obj):
    RUN.mkdir(parents=True, exist_ok=True)
    tmp = RUN / f".{name}.tmp"
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(RUN / f"{name}.json")


def load(name):
    return json.loads((RUN / f"{name}.json").read_text())


_say_lock = threading.Lock()


def say(text):
    with _say_lock:
        progress.event("pipeline", text)
        print(time.strftime("%H:%M:%S"), text, flush=True)


class Ctx:
    def __init__(self):
        self.cfg, self.base = maia.load("cuda" if torch.cuda.is_available() else "cpu")
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.calib = json.loads(CALIB.read_text())
        self.elo = {k: v for k, v in self.calib["elo"].items() if k not in self.calib["refused"]}
        self.suite = openings.load()
        self.train_ops = [o["moves"] for o in json.loads((data.ROOT / "openings_train.json").read_text())["openings"]]
        self._units = None
        self._base_scores = {}
        self._score_lock = threading.Lock()

    @property
    def units(self):
        if self._units is None:
            self._units = units.Units.load(RUN / "anchors.json", CALIB)
        return self._units

    def rated(self, games):
        r, ci = gauntlet.rate_ci(games, self.elo)
        return {"ladder": r, "ladder_ci": ci, "se": (ci[1] - ci[0]) / 3.92, "games": len(games),
                "score": sum(g["score"] for g in games) / len(games)}

    def adaptive(self, model, policy, variants, name, label, seed, n_main=200) -> dict:
        # For each variant, a short pilot against rungs spread over the ladder, then the main block against the
        # three rungs nearest the pilot's estimate, so most games are played where they say the most. A variant
        # is a dial setting, or any dict of per-game overrides (self_elo, oppo_elo, temp). All share one arena
        # per phase.
        if not isinstance(variants, dict):
            variants = {d: {"self_elo": d, "oppo_elo": d} for d in variants}
        rng = random.Random(seed)
        rungs = sorted(self.elo, key=self.elo.get)
        spread = [rungs[round(i * (len(rungs) - 1) / 3)] for i in range(4)]
        spec = lambda k, opp, op, w: {"opp": opp, "opening": op, "white": w, "seed": rng.getrandbits(31),
                                      **variants[k], "dial": k, "label": label(k)}
        todo = [spec(k, o, op, w) for k in variants for o in spread for op in self.suite[:6] for w in (True, False)]
        rng.shuffle(todo)
        by = {k: [] for k in variants}
        for g in gauntlet.run(self.cfg, model, policy, todo, seed, name, label=f"{len(variants)} settings, pilot"):
            by[g["dial"]].append(g)
        todo = []
        for k in variants:
            est = gauntlet.rate(by[k], self.elo)
            near = sorted(self.elo, key=lambda n: abs(self.elo[n] - est))[:3]
            todo += [spec(k, near[i % 3], op, w) for i, op in enumerate(self.suite[6:6 + n_main]) for w in (True, False)]
        rng.shuffle(todo)
        for g in gauntlet.run(self.cfg, model, policy, todo, seed + 1, name, label=f"{len(variants)} settings",
                              versus="the rungs nearest each"):
            by[g["dial"]].append(g)
        return by

    def base_scores(self, T):
        with self._score_lock:
            if T not in self._base_scores:
                _, raw = evals.probe(self.cfg, self.base, None, T, full=True)
                self._base_scores[T] = raw
            return self._base_scores[T]


# --- stages ------------------------------------------------------------------------------------------------

def st_snapshot(ctx):
    p = anchors.snapshot()
    return {"file": str(p.relative_to(ROOT)), "bots": json.loads(p.read_text())["bots"]}


def st_anchors(ctx):
    snap = load("snapshot")["bots"]
    pts, rows = [], []
    played = ctx.adaptive(ctx.base, {**BOT, "self_elo": 0, "oppo_elo": 0}, anchors.DIALS, "anchors",
                          lambda d: f"maia3@{d} (bot sampler)", seed=1000)
    for d in anchors.DIALS:
        games = played[d]
        r = ctx.rated(games)
        b = snap[f"maia3-79m_{d}"]["blitz"]
        sd = max(b["rd"], units.PROVISIONAL_SD) if d >= units.PROVISIONAL_FROM else b["rd"]
        pts.append((r["ladder"], r["se"], b["rating"], sd))
        rows.append({"dial": d, **r, "lichess": b["rating"], "lichess_sd": sd, "lichess_games": b["games"]})
        say(f"anchor {d}: ladder {r['ladder']:.0f} ± {1.96 * r['se']:.0f} vs Lichess {b['rating']}")
        save_games(RUN / "games" / f"anchor_{d}.json", games)
    f = units.fit(pts)
    say(f"units: {f['use']} fit, Lichess = {f[f['use']]['a']:.0f} + {f[f['use']]['b']:.3f} x ladder")
    return {"snapshot": load("snapshot")["file"], "sampler": BOT, "dials": rows, "fit": f}


def st_crosscheck(ctx):
    nets = {a: os.path.expanduser(f"~/maia1/maia-{n}.pb.gz") for a, n in zip(anchors.MAIA1, (1100, 1500, 1900))}
    if ladder.lc0_path() is None or not all(os.path.exists(p) for p in nets.values()):
        say("cross-check skipped: no lc0 binary on this machine")
        return {"skipped": "no lc0 binary"}
    snap = load("snapshot")["bots"]
    out = []
    for acct, net in nets.items():
        rng = random.Random(7)
        near = ctx.units.closest(snap[acct]["blitz"]["rating"], 3)
        jobs = [(net, near[i % 3], op, rng.getrandbits(31), white)
                for i, op in enumerate(ctx.suite[6:106]) for white in (True, False)]
        games = []
        # The games run across the engine pool like every other stage, so the rungs play under the same load.
        for g in arena.pool().imap_unordered(ladder.lc0_game, jobs):
            games.append(g)
            progress.tick("crosscheck", len(games) + 200 * len(out), 600, "games",
                          note=f"{acct} ({Path(net).stem}) vs {', '.join(near)}")
        r = ctx.rated(games)
        lich = ctx.units.lich(r["ladder"])
        out.append({"bot": acct, "net": Path(net).stem, **r, "predicted_lichess": lich,
                    "lichess": snap[acct]["blitz"]["rating"], "lichess_rd": snap[acct]["blitz"]["rd"]})
        say(f"cross-check {acct}: predicted {lich:.0f}, Lichess {snap[acct]['blitz']['rating']}")
    return {"bots": out}


def st_base(ctx):
    u, out = ctx.units, []
    played = ctx.adaptive(ctx.base, {**POLICY, "self_elo": 0, "oppo_elo": 0}, BASE_DIALS, "base",
                          lambda d: f"maia3@{d}", seed=2000)
    for d in BASE_DIALS:
        games = played[d]
        r = ctx.rated(games)
        r.update(dial=d, lichess=u.lich(r["ladder"]), lichess_ci=[u.lich(x) for x in r["ladder_ci"]],
                 lichess_ci_units=u.ci(r["ladder"], r["ladder_ci"]),
                 threefold=evals.threefold(games), book=evals.book_overlap(games))
        out.append(r)
        save_games(RUN / "games" / f"base_{d}.json", games)
        say(f"base curve {d}: plays at Lichess {r['lichess']:.0f} [{r['lichess_ci'][0]:.0f}, {r['lichess_ci'][1]:.0f}]")
    # One target where the dial plays too strong and one where it plays too weak, so the sweep has to push both
    # ways. The first run used the worst dial and 1500, and the base model was already at 1494 on the 1500 dial.
    err = lambda r: r["lichess"] - r["dial"]
    hi, lo = max(out, key=err), min(out, key=err)
    targets = [hi["dial"], lo["dial"]] if err(hi) > 0 > err(lo) else sorted({max(out, key=lambda r: abs(err(r)))["dial"], 1500})
    say(f"targets: {targets} (dial {hi['dial']} is off by {err(hi):+.0f}, dial {lo['dial']} by {err(lo):+.0f})")
    return {"policy": POLICY, "curve": out, "targets": targets}


def crossing(pts: list[tuple[float, float]], y: float) -> tuple[float, bool]:
    # Where a noisy, increasing curve of (setting, Elo) points crosses y: pooled into a monotone curve, then
    # linear between points. The flag says y was inside the measured range.
    pts = sorted(pts)
    blocks = [[x, x, v, 1] for x, v in pts]  # first x, last x, mean, count
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2]:
            a, b = blocks[i], blocks.pop(i + 1)
            blocks[i] = [a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / (a[3] + b[3]), a[3] + b[3]]
            i = max(0, i - 1)
        else:
            i += 1
    fit = [((b[0] + b[1]) / 2, b[2]) for b in blocks]
    if y <= fit[0][1]:
        return pts[0][0], y == fit[0][1]
    if y >= fit[-1][1]:
        return pts[-1][0], y == fit[-1][1]
    for (x0, v0), (x1, v1) in zip(fit, fit[1:]):
        if v0 <= y <= v1:
            return x0 + (x1 - x0) * (y - v0) / (v1 - v0), True
    return pts[-1][0], False


DIAL_GRID = range(600, 2601, 100)
TEMP_GRID = (0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5)


def st_baselines(ctx):
    # What the frozen model can do with no training, measured the way the arms are: reach Elo T, then be judged
    # on human positions at band T. If one of these lands in the band and keeps more move-match than the trained
    # arms, it is the better tool and RL is not needed for that dial.
    #   remap:  the dial setting that plays T, found on a grid of dials within 500 of T
    #   offset: T minus the base curve's mean error, one correction for every dial
    #   temp:   dial left at T, the sampling temperature that plays T
    u, base, out = ctx.units, load("base"), {}
    mean_err = sum(r["lichess"] - r["dial"] for r in base["curve"]) / len(base["curve"])
    lich = lambda games: u.lich(gauntlet.rate(games, ctx.elo))
    for T in base["targets"]:
        dials = [d for d in DIAL_GRID if abs(d - T) <= 500]
        played = ctx.adaptive(ctx.base, {**POLICY, "self_elo": 0, "oppo_elo": 0}, dials, "baselines",
                              lambda d: f"maia3@{d}", seed=6000 + T, n_main=100)
        dial_pts = [(d, lich(played[d])) for d in dials]
        d_star, d_ok = crossing(dial_pts, T)
        temps = {t: {"self_elo": T, "oppo_elo": T, "temp": t} for t in TEMP_GRID}
        played = ctx.adaptive(ctx.base, {**POLICY, "self_elo": T, "oppo_elo": T}, temps, "baselines",
                              lambda t: f"maia3@{T} temp {t}", seed=6500 + T, n_main=100)
        temp_pts = [(t, lich(played[t])) for t in TEMP_GRID]
        neg, t_ok = crossing([(-t, e) for t, e in temp_pts], T)  # hotter plays weaker
        cands = [("dial as asked", T, 1.0, True), ("remap", int(round(d_star / 10) * 10), 1.0, d_ok),
                 ("offset", int(round((T - mean_err) / 10) * 10), 1.0, True), ("temperature", T, round(-neg, 2), t_ok)]
        say(f"baselines for {T}: remap to dial {cands[1][1]}" + ("" if d_ok else " (grid never reaches it)")
            + f", offset dial {cands[2][1]}, temperature {cands[3][2]}" + ("" if t_ok else " (grid never reaches it)"))
        ref = ctx.base_scores(T)["fid_T"]
        rows_T = evals.probe_sets(T)["fid_T"]
        res = []
        for i, (method, d, t, ok) in enumerate(cands):
            pol = {**POLICY, "self_elo": d, "oppo_elo": d, "temp": t}
            games = gauntlet.play(ctx.cfg, ctx.base, pol, u.closest(T, 3), ctx.suite[:EVAL_OPENINGS] * 2, seed=6900 + i,
                                  name="baselines", elo=ctx.elo, label=f"{method}: dial {d}, temp {t}", cycle=True)
            r = ctx.rated(games)
            raw = evals.score(ctx.cfg, ctx.base, rows_T, self_elo=d, temp=t)
            fid = evals.summary(raw)
            fid["top1_vs_base_ci"] = [100 * x for x in evals.paired_ci(raw["hit"], ref["hit"])]
            fid["sce_vs_base_ci"] = evals.paired_ci(raw["nll"], ref["nll"])
            row = {"method": method, "dial": d, "temp": t, "reaches": ok, "elo": u.lich(r["ladder"]),
                   "elo_ci": [u.lich(x) for x in r["ladder_ci"]], "elo_ci_units": u.ci(r["ladder"], r["ladder_ci"]),
                   "ladder": r, "fidelity": fid, "humanlike": evals.humanlike(evals.moves_of(games, HUMANLIKE_MOVES))}
            res.append(row)
            save_games(RUN / "games" / f"baseline_{T}_{method.replace(' ', '_')}.json", games)
            say(f"baseline {method} for {T}: dial {d}, temp {t}: Elo {row['elo']:.0f} [{row['elo_ci'][0]:.0f}, "
                f"{row['elo_ci'][1]:.0f}], top-1 {fid['top1']:.1f}% ({fid['top1_vs_base_ci'][0]:+.1f} to "
                f"{fid['top1_vs_base_ci'][1]:+.1f} vs the dial as asked)")
        out[T] = {"dial_grid": dial_pts, "temp_grid": temp_pts, "mean_error": mean_err, "methods": res}
    return out


def make_eval(ctx, T):
    base_raw = None
    prev = {}

    def on_eval(tr: Trainer, tag: str) -> dict:
        nonlocal base_raw
        if base_raw is None:
            base_raw = ctx.base_scores(T)
        u = ctx.units
        pol = {**POLICY, "self_elo": T, "oppo_elo": T}
        games = gauntlet.play(ctx.cfg, tr.ema, pol, u.closest(tr.goal, 3), ctx.suite[:EVAL_OPENINGS], seed=3000,
                              name=f"{tr.name}:eval", elo=ctx.elo, label=f"policy@{T} ({tag})", cycle=True)
        r = ctx.rated(games)
        lich, ci = u.lich(r["ladder"]), [u.lich(x) for x in r["ladder_ci"]]
        pr, _ = evals.probe(ctx.cfg, tr.ema, base_raw, T)  # includes the validation slice the band-aware pick uses
        last = tr.st["log"][-1] if tr.st["log"] else {}
        ev = {"tag": tag, "update": tr.st["update"], "games_trained": tr.st["games"], "elo": lich, "elo_ci": ci,
              "ladder": r, "probe": pr, "entropy": last.get("entropy"), "kl_base": last.get("kl_base"),
              "judge": last.get("judge"), "threefold": evals.threefold(games), "book": evals.book_overlap(games)}
        save_games(tr.out / f"eval_{tag}.games.json", games)
        # The judge's score rising while the move match on real humans at the band falls is the hack to catch.
        band = f"band_{T}" if f"band_{T}" in pr else "calib"
        if prev and ev["judge"] is not None and prev.get("judge") is not None:
            # Minimum sizes keep eval-to-eval noise out: 0.02 nats on the judge, 0.01 on human SCE.
            if ev["judge"] - prev["judge"] > 0.02 and pr[band]["sce"] - prev["probe"][band]["sce"] > 0.01:
                ev["judge_gap"] = {"judge": [prev["judge"], ev["judge"]],
                                   "human_sce": [prev["probe"][band]["sce"], pr[band]["sce"]]}
        prev.clear()
        prev.update(ev)
        # The rating of training games drifts from what a gauntlet of the EMA measures. Half of the gap seen
        # here goes into the correction, so one noisy eval cannot swing it.
        raw = tr.st.get("est_raw")
        if raw is not None:
            tr.st["bias"] = 0.5 * tr.st["bias"] + 0.5 * (lich - raw)
            ev.update(est_raw=raw, bias=tr.st["bias"], weight=tr.st["sign"])
        say(f"{tr.name} @{tag}: Elo {lich:.0f} [{ci[0]:.0f}, {ci[1]:.0f}] vs goal {tr.goal:.0f}, top-1 {pr[band]['top1']:.1f}%"
            + (f", training games read {raw:.0f}, outcome weight {tr.st['sign']:+.2f}" if raw is not None else ""))
        tr.report(f"checkpoint {tag} · Elo {lich:.0f} · top-1 {pr[band]['top1']:.1f}%",
                  stats={"elo": f"{lich:.0f}", "top1": f"{pr[band]['top1']:.1f}%"})
        return ev
    return on_eval


def st_smoke(ctx):
    out = RUN / "smoke_run"
    shutil.rmtree(out, ignore_errors=True)
    T = 1500
    start = next(r["lichess"] for r in load("base")["curve"] if r["dial"] == T)
    mk = lambda: Trainer(ctx.cfg, ctx.base, out, T, start + 150, start, units=ctx.units, games=SMOKE_GAMES, name="smoke")
    tr = mk()
    st = tr.run(ctx.train_ops, lambda tr, tag: {"tag": tag})
    log = [x for x in st["log"] if "loss" in x]
    ok_loss = bool(log) and all(map(lambda x: all(isinstance(v, (int, float)) and v == v for v in x.values()), log))
    tr2 = mk()
    resumed = tr2.st["update"] == st["update"] and tr2.st["games"] == st["games"]
    # The goal is 150 Elo above the base model, so the outcome weight has to have grown toward winning.
    steered = st.get("est") is not None and st["nu"] > 0
    res = {"updates": st["update"], "games": st["games"], "finite_loss": ok_loss, "steered": steered,
           "kl_logged": all("kl_base" in x for x in log), "resume_ok": resumed, "log": log}
    if not (ok_loss and res["kl_logged"] and resumed and steered and st["update"] >= 2):
        raise RuntimeError(f"smoke run failed: { {k: v for k, v in res.items() if k != 'log'} }")
    shutil.rmtree(out, ignore_errors=True)
    say(f"smoke run passed: {st['update']} updates, finite loss, KL logged, resume ok, outcome weight {st['sign']:+.2f}")
    return res


def arm_name(T, frac):
    return f"T{T}_goal{round(100 * frac)}"


def held(log: list[dict], goal: float) -> dict:
    # How well the last third of training stayed on the goal, from the running rating at each update.
    v = [x["est"] for x in log if "est" in x]
    v = v[-max(1, len(v) // 3):]
    if not v:
        return {}
    m = sum(v) / len(v)
    return {"updates": len(v), "mean": m, "sd": (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5,
            "in_band": sum(abs(x - goal) <= BAND for x in v) / len(v)}


def finish_arm(ctx, T, tr: Trainer, frac: float, start: float) -> dict:
    # The checkpoint reported is the last one. The first two runs picked the evaluated checkpoint nearest the
    # target, which flatters a policy that swings through it. It gets a fresh 800-game gauntlet and the 10k
    # fidelity set, then the dial test and the mistake analysis.
    u = ctx.units
    pol = {**POLICY, "self_elo": T, "oppo_elo": T}
    confirm_games = 4 * len(ctx.suite[:EVAL_OPENINGS])
    sensitivity_games = 6 * len(ctx.suite[:SENS_OPENINGS])
    final_total, final_steps = 2 * confirm_games + sensitivity_games, 2
    stats = lambda step, before, active=True: {"final_active": active, "final_before": before, "final_total": final_total,
                                               "final_step": step, "final_steps": final_steps}
    tr.report("final check 1/2 · fresh rated games for the last checkpoint", force=True, stats=stats(1, 0))
    model = maia.load(ctx.cfg.device)[1].eval()
    model.load_state_dict(torch.load(tr.out / "ema_final.pt", map_location=ctx.cfg.device))
    games = gauntlet.play(ctx.cfg, model, pol, u.closest(tr.goal, 3), ctx.suite[:EVAL_OPENINGS] * 2, seed=4000,
                          name=f"{tr.name}:confirm", elo=ctx.elo, label=f"policy@{T} final", cycle=True)
    r = ctx.rated(games)
    pr, _ = evals.probe(ctx.cfg, model, ctx.base_scores(T), T, full=True)
    final = {"tag": "final", "elo": u.lich(r["ladder"]), "elo_ci": [u.lich(x) for x in r["ladder_ci"]],
             "elo_ci_units": u.ci(r["ladder"], r["ladder_ci"]), "ladder": r, "fidelity": pr.get("fid_T"), "probe": pr}
    save_games(tr.out / "confirm_final.games.json", games)
    torch.save(model.state_dict(), tr.out / "final.pt")
    # A second, selected checkpoint: of the evals that measured inside the band, the one that matches humans
    # best on the validation slice, which shares no game or month with the 10k set it is then reported on. In the second run the outcome-only arm kept losing move-match after it had reached
    # the band (57.3, 55.2, 54.0% at 1900), so an earlier in-band checkpoint was the better model. Its Elo comes
    # from a fresh gauntlet, since "inside the band" was itself read off a noisy eval.
    inband = [e for e in tr.st["evals"] if "elo" in e and abs(e["elo"] - tr.goal) <= BAND and e["probe"].get("val_T")]
    pick = max(inband, key=lambda e: e["probe"]["val_T"]["top1"], default=None)
    band_pick = None
    if pick is not None and pick["tag"] == "final":
        band_pick = {**final, "eval_elo": pick["elo"]}
    elif pick is not None:
        m2 = maia.load(ctx.cfg.device)[1].eval()
        m2.load_state_dict(torch.load(tr.out / f"ema_{pick['tag']}.pt", map_location=ctx.cfg.device))
        g2 = gauntlet.play(ctx.cfg, m2, pol, u.closest(tr.goal, 3), ctx.suite[:EVAL_OPENINGS] * 2, seed=4001,
                           name=f"{tr.name}:confirm", elo=ctx.elo, label=f"policy@{T} checkpoint {pick['tag']}", cycle=True)
        r2 = ctx.rated(g2)
        p2, _ = evals.probe(ctx.cfg, m2, ctx.base_scores(T), T, full=True)
        band_pick = {"tag": pick["tag"], "eval_elo": pick["elo"], "elo": u.lich(r2["ladder"]),
                     "elo_ci": [u.lich(x) for x in r2["ladder_ci"]], "elo_ci_units": u.ci(r2["ladder"], r2["ladder_ci"]),
                     "ladder": r2, "fidelity": p2.get("fid_T")}
        torch.save(m2.state_dict(), tr.out / "band_pick.pt")
        del m2
    # Conditioning sensitivity: the same openings and seeds at three dial settings.
    tr.report(f"final check 2/2 · testing rating control at {T - 200}, {T}, {T + 200}", force=True,
              stats=stats(2, confirm_games))
    rng, near = random.Random(5000), u.closest(tr.goal, 3)
    todo = [{"opp": near[i % 3], "opening": op, "white": w, "seed": rng.getrandbits(31), "self_elo": d, "oppo_elo": d,
             "dial": d, "label": f"final@{d}"}
            for d in (T - 200, T, T + 200) for i, op in enumerate(ctx.suite[:SENS_OPENINGS]) for w in (True, False)]
    rng.shuffle(todo)
    played = gauntlet.run(ctx.cfg, model, pol, todo, 5000, f"{tr.name}:sensitivity", label="final, dial moved ±200",
                          versus=", ".join(near))
    sens = {}
    for d in (T - 200, T, T + 200):
        r = ctx.rated([g for g in played if g["dial"] == d])
        sens[d] = {"elo": u.lich(r["ladder"]), "elo_ci": [u.lich(x) for x in r["ladder_ci"]], **r}
    hl = evals.humanlike(evals.moves_of(games, HUMANLIKE_MOVES))
    tr.report("final checks complete", force=True, stats=stats(2, final_total, False))
    del model
    torch.cuda.empty_cache()
    log = tr.st["log"]
    return {"T": T, "goal": tr.goal, "start": start, "frac": frac, "evals": tr.st["evals"], "final": final,
            "band_pick": band_pick,
            "sensitivity": sens, "humanlike": hl, "recoveries": tr.st["recoveries"], "held": held(log, tr.goal),
            "weight": tr.st["sign"], "bias": tr.st["bias"], "train_log": log,
            "loss": {"kp": KP, "eta": ETA, "pull": PULL, "band": BAND, "ctrl_games": CTRL_GAMES}}


class SweepProgress:
    def __init__(self, targets):
        self.targets = targets
        self.total = len(targets) * len(GOALS) * ARM_GAMES
        self.parts, self.trainers = {}, {}
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()

    def add(self, tr):
        self.trainers[tr.name] = tr
        self.update()

    def complete(self, name):
        self.parts[name] = ARM_GAMES
        self.trainers.pop(name, None)
        self.update()

    def update(self):
        done = sum(self.parts.values()) + sum(min(tr.st["games"], ARM_GAMES) for tr in self.trainers.values())
        active = sum(tr.st["games"] < ARM_GAMES for tr in self.trainers.values())
        total_arms = len(self.targets) * len(GOALS)
        progress.tick("sweep", done, self.total, "games",
                      note=f"{len(self.parts)}/{total_arms} arms complete · {active} active")

    def poll(self):
        while not self.stopping.wait(1):
            self.update()

    def close(self):
        self.stopping.set()
        self.thread.join()
        self.update()


def st_sweep(ctx):
    base = load("base")
    targets = base["targets"]
    results = {}
    tracker = SweepProgress(targets)
    try:
        for T in targets:
            start = next(r["lichess"] for r in base["curve"] if r["dial"] == T)
            goal = lambda f: start + f * (T - start)
            pending = []
            for f in GOALS:
                name = arm_name(T, f)
                if done(f"arm_{name}"):
                    results[name] = load(f"arm_{name}")
                    tracker.complete(name)
                else:
                    pending.append(f)

            def run(f):
                name = arm_name(T, f)
                say(f"training {name}: dial {T}, from {start:.0f} to {goal(f):.0f}")
                tr = Trainer(ctx.cfg, ctx.base, ROOT / "runs" / "arms" / name, T, goal(f), start, units=ctx.units,
                             games=ARM_GAMES, seed=zlib.crc32(name.encode()) % 10_000, name=name)
                tracker.add(tr)
                tr.run(ctx.train_ops, make_eval(ctx, T))
                return name, f, tr

            with concurrent.futures.ThreadPoolExecutor(max_workers=SWEEP_WORKERS) as pool:
                trained = list(pool.map(run, pending))
            for name, f, tr in trained:
                res = finish_arm(ctx, T, tr, f, start)
                save(f"arm_{name}", res)
                for ck in tr.out.glob("*.pt"):  # about 2.5 GB an arm otherwise, and only final.pt is used again
                    if ck.name not in ("final.pt", "band_pick.pt"):
                        ck.unlink()
                tracker.complete(name)
                results[name] = res
                fi, h = res["final"], res["held"]
                say(f"{name} final: Elo {fi['elo']:.0f} [{fi['elo_ci'][0]:.0f}, {fi['elo_ci'][1]:.0f}] vs goal {tr.goal:.0f}, "
                    f"top-1 {fi['fidelity']['top1']:.1f}%, in band {h.get('in_band', 0):.0%} of the last third")
        # Human-likeness of the untouched model and of real humans at each target, for the comparison.
        ref, checks = {}, []
        for T in targets:
            bg = json.loads((RUN / "games" / f"base_{T}.json").read_text())
            checks.extend([(T, "base model", evals.moves_of(bg, HUMANLIKE_MOVES)), (T, "human games", evals.human_moves(T, HUMANLIKE_MOVES))])
        total = sum(len(moves) for _, _, moves in checks)
        progress.tick("comparison", 0, total, "positions", force=True, note="preparing final comparison")
        completed = 0
        for T, kind, moves in checks:
            def update(n, start=completed, T=T, kind=kind):
                progress.tick("comparison", start + n, total, "positions",
                              note=f"{kind} @ {T} · Stockfish mistake analysis")
            ref.setdefault(T, {})[kind.split()[0]] = evals.humanlike(moves, on_progress=update)
            completed += len(moves)
            progress.tick("comparison", completed, total, "positions", force=True,
                          note=f"{kind} @ {T} · analysis complete")
        return {"targets": targets, "arms": list(results), "humanlike_reference": ref}
    finally:
        tracker.close()


def st_report(ctx):
    from levelrl import report
    return report.write(RUN, RES)


# --- plumbing ----------------------------------------------------------------------------------------------

def save_games(path: Path, games):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{k: v for k, v in g.items() if k != "rec"} for g in games]))


def pgn(g: dict, white: str, black: str) -> str:
    b = chess.Board()
    for m in g["moves"]:
        b.push_uci(m)
    game = chess.pgn.Game.from_board(b)
    game.headers.update({"White": white, "Black": black, "Event": "LevelRL"})
    return str(game)


def main():
    t0 = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    man = ROOT / "MANIFEST.json"  # written by scripts/deploy.sh
    if not (RUN / "manifest.json").exists():
        (RUN / "manifest.json").write_text(man.read_text() if man.exists() else json.dumps({"commit": None}))
    say(f"code: {json.loads((RUN / 'manifest.json').read_text()).get('commit') or 'unknown commit, not deployed with scripts/deploy.sh'}")
    ctx = Ctx()
    try:
        for i, name in enumerate(STAGES):
            progress.tick("pipeline", i, len(STAGES), "stages", force=True, note=f"stage: {name}")
            if done(name):
                continue
            for attempt in range(3):
                try:
                    save(name, globals()[f"st_{name}"](ctx))
                    break
                except Exception:
                    tb = traceback.format_exc()
                    (RUN / f"{name}.error.txt").write_text(tb)
                    say(f"{name} failed (attempt {attempt + 1}): {tb.strip().splitlines()[-1][:120]}")
                    torch.cuda.empty_cache()
            else:
                if name in OPTIONAL:
                    save(name, {"skipped": "failed three times, see the error file"})
                    continue
                say(f"pipeline stopped at {name}")
                raise SystemExit(1)
        progress.tick("pipeline", len(STAGES), len(STAGES), "stages", force=True, note="all stages done")
        say(f"pipeline finished in {(time.time() - t0) / 3600:.1f} h, results in results/")
    finally:
        arena.close_pool()
        ladder.close_engines()


if __name__ == "__main__":
    main()
