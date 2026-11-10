from __future__ import annotations

import json
import os
import random
import shutil
import time
import traceback
import zlib
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

from levelrl import anchors, arena, data, evals, gauntlet, ladder, maia, openings, progress, units
from levelrl.train import CTRL_GAMES, KL_COEF, PULL, Trainer, pull

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
ARMS = ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.75, 0.25), (0.25, 0.75))  # endpoints first
ARM_GAMES = 3000
SMOKE_GAMES = 96
EVAL_OPENINGS = 200
SENS_OPENINGS = 100
STAGES = ("snapshot", "anchors", "crosscheck", "base", "smoke", "sweep", "report")
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


def say(text):
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

    @property
    def units(self):
        if self._units is None:
            self._units = units.Units.load(RUN / "anchors.json", CALIB)
        return self._units

    def rated(self, games):
        r, ci = gauntlet.rate_ci(games, self.elo)
        return {"ladder": r, "ladder_ci": ci, "se": (ci[1] - ci[0]) / 3.92, "games": len(games),
                "score": sum(g["score"] for g in games) / len(games)}

    def adaptive(self, model, policy, dials, name, label, seed, n_main=200) -> dict[int, list]:
        # For each dial, a short pilot against rungs spread over the ladder, then the main block against the three
        # rungs nearest the pilot's estimate, so most games are played where they say the most. All dials share
        # one arena per phase.
        rng = random.Random(seed)
        rungs = sorted(self.elo, key=self.elo.get)
        spread = [rungs[round(i * (len(rungs) - 1) / 3)] for i in range(4)]
        spec = lambda d, opp, op, w: {"opp": opp, "opening": op, "white": w, "seed": rng.getrandbits(31),
                                      "self_elo": d, "oppo_elo": d, "dial": d, "label": label(d)}
        todo = [spec(d, o, op, w) for d in dials for o in spread for op in self.suite[:6] for w in (True, False)]
        rng.shuffle(todo)
        by = {d: [] for d in dials}
        for g in gauntlet.run(self.cfg, model, policy, todo, seed, name, label=f"{len(dials)} dials, pilot"):
            by[g["dial"]].append(g)
        todo = []
        for d in dials:
            est = gauntlet.rate(by[d], self.elo)
            near = sorted(self.elo, key=lambda n: abs(self.elo[n] - est))[:3]
            todo += [spec(d, near[i % 3], op, w) for i, op in enumerate(self.suite[6:6 + n_main]) for w in (True, False)]
        rng.shuffle(todo)
        for g in gauntlet.run(self.cfg, model, policy, todo, seed + 1, name, label=f"{len(dials)} dials",
                              versus="the rungs nearest each"):
            by[g["dial"]].append(g)
        return by

    def base_scores(self, T):
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


def make_eval(ctx, T):
    base_raw = None
    prev = {}

    def on_eval(tr: Trainer, tag: str) -> dict:
        nonlocal base_raw
        if base_raw is None:
            base_raw = ctx.base_scores(T)
        u = ctx.units
        pol = {**POLICY, "self_elo": T, "oppo_elo": T}
        games = gauntlet.play(ctx.cfg, tr.ema, pol, u.closest(T, 3), ctx.suite[:EVAL_OPENINGS], seed=3000,
                              name=tr.name, elo=ctx.elo, label=f"policy@{T} ({tag})", cycle=True)
        r = ctx.rated(games)
        lich, ci = u.lich(r["ladder"]), [u.lich(x) for x in r["ladder_ci"]]
        pr, _ = evals.probe(ctx.cfg, tr.ema, base_raw, T)
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
        est = tr.st.get("est")
        say(f"{tr.name} @{tag}: Elo {lich:.0f} [{ci[0]:.0f}, {ci[1]:.0f}] vs T {T}, top-1 {pr[band]['top1']:.1f}%"
            + (f", running estimate {est:.0f}, pull {tr.st['sign']:+.2f}" if est is not None else ""))
        return ev
    return on_eval


def st_smoke(ctx):
    out = RUN / "smoke_run"
    shutil.rmtree(out, ignore_errors=True)
    T = 1500
    tr = Trainer(ctx.cfg, ctx.base, out, T, 0.5, 0.5, sign=1, units=ctx.units, games=SMOKE_GAMES, name="smoke")
    st = tr.run(ctx.train_ops, lambda tr, tag: {"tag": tag})
    log = [x for x in st["log"] if "loss" in x]
    ok_loss = bool(log) and all(map(lambda x: all(isinstance(v, (int, float)) and v == v for v in x.values()), log))
    tr2 = Trainer(ctx.cfg, ctx.base, out, T, 0.5, 0.5, sign=1, units=ctx.units, games=SMOKE_GAMES, name="smoke")
    resumed = tr2.st["update"] == st["update"] and tr2.st["games"] == st["games"]
    res = {"updates": st["update"], "games": st["games"], "finite_loss": ok_loss,
           "kl_logged": all("kl_base" in x for x in log), "resume_ok": resumed, "log": log}
    if not (ok_loss and res["kl_logged"] and resumed and st["update"] >= 2):
        raise RuntimeError(f"smoke run failed: {res}")
    shutil.rmtree(out, ignore_errors=True)
    say(f"smoke run passed: {st['update']} updates, finite loss, KL logged, resume ok")
    return res


def arm_name(T, wo, wf):
    return f"T{T}_out{wo}_fid{wf}"


def finish_arm(ctx, T, tr: Trainer, base_curve) -> dict:
    # Winner's-curse guard: the three evaluated checkpoints nearest T get a fresh 800-game gauntlet and the 10k
    # fidelity set, and the final pick is made on those numbers only.
    u = ctx.units
    evs = [e for e in tr.st["evals"] if "elo" in e]
    top = sorted(evs, key=lambda e: abs(e["elo"] - T))[:3]
    pol = {**POLICY, "self_elo": T, "oppo_elo": T}
    conf = []
    model = maia.load(ctx.cfg.device)[1].eval()
    for e in top:
        model.load_state_dict(torch.load(tr.out / f"ema_{e['tag']}.pt", map_location=ctx.cfg.device))
        games = gauntlet.play(ctx.cfg, model, pol, u.closest(T, 3), ctx.suite[:EVAL_OPENINGS] * 2, seed=4000,
                              name=tr.name, elo=ctx.elo, label=f"policy@{T} confirm {e['tag']}", cycle=True)
        r = ctx.rated(games)
        pr, _ = evals.probe(ctx.cfg, model, ctx.base_scores(T), T, full=True)
        conf.append({"tag": e["tag"], "elo": u.lich(r["ladder"]), "elo_ci": [u.lich(x) for x in r["ladder_ci"]],
                     "ladder": r, "fidelity": pr.get("fid_T"), "probe": pr})
        save_games(tr.out / f"confirm_{e['tag']}.games.json", games)
    final = min(conf, key=lambda c: abs(c["elo"] - T))
    model.load_state_dict(torch.load(tr.out / f"ema_{final['tag']}.pt", map_location=ctx.cfg.device))
    torch.save(model.state_dict(), tr.out / "final.pt")
    # Conditioning sensitivity: the same openings and seeds at three dial settings.
    rng, near = random.Random(5000), u.closest(T, 3)
    todo = [{"opp": near[i % 3], "opening": op, "white": w, "seed": rng.getrandbits(31), "self_elo": d, "oppo_elo": d,
             "dial": d, "label": f"final@{d}"}
            for d in (T - 200, T, T + 200) for i, op in enumerate(ctx.suite[:SENS_OPENINGS]) for w in (True, False)]
    rng.shuffle(todo)
    played = gauntlet.run(ctx.cfg, model, pol, todo, 5000, tr.name, label="final, dial moved ±200",
                          versus=", ".join(near))
    sens = {}
    for d in (T - 200, T, T + 200):
        r = ctx.rated([g for g in played if g["dial"] == d])
        sens[d] = {"elo": u.lich(r["ladder"]), "elo_ci": [u.lich(x) for x in r["ladder_ci"]], **r}
    fgames = json.loads((tr.out / f"confirm_{final['tag']}.games.json").read_text())
    hl = evals.humanlike(evals.moves_of(fgames, 8000))
    del model
    torch.cuda.empty_cache()
    return {"T": T, "w_out": tr.w_out, "w_fid": tr.w_fid, "evals": tr.st["evals"], "confirm": conf,
            "final": final, "sensitivity": sens, "humanlike": hl, "recoveries": tr.st["recoveries"],
            "train_log": tr.st["log"], "loss": {"kl_coef": KL_COEF, "pull": PULL, "ctrl_games": CTRL_GAMES}}


def st_sweep(ctx):
    base = load("base")
    targets = base["targets"]
    results = {}
    for T in targets:
        bc = next((r for r in base["curve"] if r["dial"] == T), None)
        sign = 1.0 if bc is None else pull(T, bc["lichess"])
        for wo, wf in ARMS:
            name = arm_name(T, wo, wf)
            if done(f"arm_{name}"):
                results[name] = load(f"arm_{name}")
                continue
            say(f"training {name}")
            tr = Trainer(ctx.cfg, ctx.base, ROOT / "runs" / "arms" / name, T, wo, wf, sign=sign, units=ctx.units,
                         games=ARM_GAMES, seed=zlib.crc32(name.encode()) % 10_000, name=name)
            tr.run(ctx.train_ops, make_eval(ctx, T))
            res = finish_arm(ctx, T, tr, base)
            save(f"arm_{name}", res)
            results[name] = res
            f = res["final"]
            say(f"{name} final: Elo {f['elo']:.0f} [{f['elo_ci'][0]:.0f}, {f['elo_ci'][1]:.0f}], "
                f"fidelity top-1 {f['fidelity']['top1']:.1f}%")
    # Human-likeness of the untouched model and of real humans at each target, for the comparison.
    ref = {}
    for T in targets:
        bg = json.loads((RUN / "games" / f"base_{T}.json").read_text())
        ref[T] = {"base": evals.humanlike(evals.moves_of(bg, 8000)), "human": evals.humanlike(evals.human_moves(T))}
    return {"targets": targets, "arms": list(results), "humanlike_reference": ref}


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
