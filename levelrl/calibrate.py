from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from levelrl import arena, data, evals, gauntlet, pipeline, progress

DIAL_GRID = tuple(range(600, 2601, 100))
TEMP_GRID = (0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5)
BAND = 50
SCAN_MAIN = 48
VALIDATION_OPENINGS = 50
HUMANLIKE_MOVES = 16_000
VERSION = 1


def has_fidelity(T: int) -> bool:
    return (data.ROOT / "positions" / f"fid_{T}.jsonl.gz").exists()


def candidates(T: int, dial_pts: list, temp_pts: list) -> list[dict]:
    d_star, d_ok = pipeline.crossing(dial_pts, T)
    neg, t_ok = pipeline.crossing([(-t, e) for t, e in temp_pts], T)
    t_star = -neg
    d_mid = int(round((T + d_star) / 20) * 10)
    t_mid = round((1 + t_star) / 2, 3)
    return [
        {"method": "dial as asked", "dial": T, "temp": 1.0, "reaches": True},
        {"method": "remap", "dial": int(round(d_star / 10) * 10), "temp": 1.0, "reaches": d_ok},
        {"method": "temperature", "dial": T, "temp": round(t_star, 3), "reaches": t_ok},
        {"method": "combined", "dial": d_mid, "temp": t_mid, "reaches": d_ok and t_ok},
    ]


def choose(rows: list[dict], T: int) -> dict:
    inside = [r for r in rows if abs(r["validation_elo"] - T) <= BAND]
    if inside:
        return min(inside, key=lambda r: (r.get("validation_sce", math.inf), abs(r["validation_elo"] - T)))
    return min(rows, key=lambda r: abs(r["validation_elo"] - T))


def pareto(rows: list[dict], T: int) -> list[dict]:
    rows = [r for r in rows if "validation_elo" in r]
    out = []
    for r in rows:
        err = abs(r["validation_elo"] - T)
        sce = r.get("validation_sce", math.inf)
        dominated = any(abs(q["validation_elo"] - T) <= err and q.get("validation_sce", math.inf) <= sce
                        and (abs(q["validation_elo"] - T) < err or q.get("validation_sce", math.inf) < sce)
                        for q in rows)
        if not dominated:
            out.append(r)
    return out


def mapping(rows: list[dict]) -> list[dict]:
    return [{"target": r["target"], "method": r["method"], "dial": r["dial"], "temp": r["temp"],
             "measured_elo": r["elo"], "elo_ci": r["elo_ci"], "elo_ci_units": r["elo_ci_units"],
             "strength_in_band": abs(r["elo"] - r["target"]) <= BAND, "validated": True}
            for r in sorted(rows, key=lambda r: r["target"])]


def setting_at(rows: list[dict], target: int) -> dict:
    if not rows:
        raise ValueError("an empty calibration map has no settings")
    rows = sorted(rows, key=lambda r: r["target"])
    nearest = min(rows, key=lambda r: (abs(r["target"] - target), r["target"]))
    return {"method": nearest["method"], "dial": nearest["dial"], "temp": nearest["temp"],
            "nearest_target": nearest["target"], "interpolated": False,
            "extrapolated": target < rows[0]["target"] or target > rows[-1]["target"]}


def _atomic(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _rated(ctx, games, T):
    r = ctx.rated(games)
    return {"elo": ctx.units.lich(r["ladder"]), "elo_ci": [ctx.units.lich(x) for x in r["ladder_ci"]],
            "elo_ci_units": ctx.units.ci(r["ladder"], r["ladder_ci"]), "ladder": r}


_VALIDATION_REF, _FINAL_REF = {}, {}


def _fid(ctx, T, dial, temp, final):
    split = "fid_T" if final else "val_T"
    rows = evals.probe_sets(T, bands=(T,))[split]
    raw = evals.score(ctx.cfg, ctx.base, rows, self_elo=dial, temp=temp)
    refs = _FINAL_REF if final else _VALIDATION_REF
    if T not in refs:
        refs[T] = evals.score(ctx.cfg, ctx.base, rows, self_elo=T)
    ref = refs[T]
    out = evals.summary(raw)
    out["top1_vs_base_ci"] = [100 * x for x in evals.paired_ci(raw["hit"], ref["hit"])]
    out["sce_vs_base_ci"] = evals.paired_ci(raw["nll"], ref["nll"])
    return out


def _reference(ctx, T, state, run_dir=None):
    key = str(T)
    if key not in state["references"]:
        path = pipeline.RUN / "games" / f"base_{T}.json"
        if not path.exists() and run_dir is not None:
            path = Path(run_dir) / "games" / f"base_{T}.json"
        bg = json.loads(path.read_text())
        state["references"][key] = {
            "base": evals.humanlike(evals.moves_of(bg, HUMANLIKE_MOVES), job="calibration:humanlike",
                                    what=f"base model @ {T}"),
            "human": evals.humanlike(evals.human_moves(T, HUMANLIKE_MOVES), job="calibration:humanlike",
                                     what=f"humans @ {T}"),
        }
    return state["references"][key]


def selected_rows(state):
    rows = []
    for T, st in state["targets"].items():
        if "final" not in st:
            continue
        row = {"target": int(T), **st["selected"], **st["final"]}
        rows.append(row)
    return sorted(rows, key=lambda r: r["target"])


def write_outputs(state, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    _atomic(out / "calibration.json", state)
    rows = selected_rows(state)
    fit = mapping(rows) if rows else []
    settings = []
    if fit:
        for T in range(fit[0]["target"], fit[-1]["target"] + 1, 100):
            settings.append({"target": T, **setting_at(fit, T)})
    _atomic(out / "calibration_map.json", {
        "units": "estimated Lichess blitz strength from the current fixed-node ladder calibration",
        "interpolation": "exact measured settings only; unmeasured targets use the nearest measured configuration without claiming validation",
        "validated": fit,
        "settings": settings,
    })
    lines = ["# Maia-3 full-dial calibration", "",
             "Settings are selected on validation games and validation human positions. Final games use fresh openings; "
             "the 10,000-position fidelity set is only read after selection. Ratings are estimated Lichess blitz strength, "
             "not official Lichess ratings.", "",
             "| target | selected method | dial | temperature | measured strength | 95% CI in run | with units | error | top-1 | log-loss |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        fid = r.get("fidelity") or {}
        lines.append(f"| {r['target']} | {r['method']} | {r['dial']} | {r['temp']:g} | {r['elo']:.0f} | "
                     f"{r['elo_ci'][0]:.0f} to {r['elo_ci'][1]:.0f} | {r['elo_ci_units'][0]:.0f} to {r['elo_ci_units'][1]:.0f} | "
                     f"{r['elo'] - r['target']:+.0f} | {fid.get('top1', float('nan')):.1f} | {fid.get('sce', float('nan')):.3f} |")
    available = [r["target"] for r in rows if r.get("fidelity")]
    lines += ["", "Human fidelity is available at " + ", ".join(map(str, available)) +
              "; other rows report strength only.", ""]
    if any(r.get("humanlike") for r in rows):
        def pm(p, n):
            return f"{100 * p:.1f} ± {196 * (p * (1 - p) / max(1, n)) ** 0.5:.1f}"

        def cells(h):
            ph = h["by_phase"]
            return [pm(ph[k]["mistake_rate"], ph[k]["n"]) if k in ph else "n/a"
                    for k in ("opening", "middlegame", "endgame")]

        lines += ["## Sampled-game human-likeness", "",
                  "Percent of moves, with normal-approximation 95% intervals. These intervals do not include game-to-game correlation.",
                  "", "| target | player | blunders | opening mistakes | middlegame | endgame |", "|---|---|---|---|---|---|"]
        for r in rows:
            if not r.get("humanlike"):
                continue
            players = [(r["method"], r["humanlike"])]
            ref = r.get("reference") or {}
            players += [(name, ref[name]) for name in ("base", "human") if name in ref]
            for name, h in players:
                lines.append(f"| {r['target']} | {name} | {pm(h['blunder_rate'], h['n'])} | " +
                             " | ".join(cells(h)) + " |")
        lines.append("")
    lines += ["## Validation Pareto frontiers", ""]
    for T, st in state["targets"].items():
        if "candidates" not in st:
            continue
        lines += [f"### {T}", "", "| method | dial | temperature | validation strength | validation log-loss | selected |",
                  "|---|---|---:|---:|---:|---|"]
        chosen = st.get("selected", {})
        for r in pareto(st["candidates"], int(T)):
            mark = "yes" if (r["method"], r["dial"], r["temp"]) == (chosen.get("method"), chosen.get("dial"), chosen.get("temp")) else ""
            lines.append(f"| {r['method']} | {r['dial']} | {r['temp']:g} | {r['validation_elo']:.0f} | "
                         f"{r.get('validation_sce', float('nan')):.3f} | {mark} |")
        lines.append("")
    (out / "calibration.md").write_text("\n".join(lines) + "\n")
    from levelrl import coverage
    coverage.write_readiness(state, out)


def plot_outputs(state, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = selected_rows(state)
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].errorbar([r["target"] for r in rows], [r["elo"] for r in rows],
                         yerr=[[r["elo"] - r["elo_ci"][0] for r in rows], [r["elo_ci"][1] - r["elo"] for r in rows]],
                         fmt="o-", capsize=3)
        lo, hi = rows[0]["target"], rows[-1]["target"]
        axes[0].plot([lo, hi], [lo, hi], "k:", label="target = measured")
        axes[0].set(xlabel="requested Lichess blitz strength", ylabel="measured estimate")
        axes[0].legend()
        axes[1].plot([r["target"] for r in rows], [r["dial"] for r in rows], "o-")
        axes[1].set(xlabel="target", ylabel="Maia dial")
        axes[2].plot([r["target"] for r in rows], [r["temp"] for r in rows], "o-")
        axes[2].set(xlabel="target", ylabel="temperature")
        fig.tight_layout()
        fig.savefig(out / "calibration_map.png", dpi=140)
        plt.close(fig)
    for T, st in state["targets"].items():
        rows = [r for r in st.get("candidates", []) if "validation_sce" in r]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for r in rows:
            ax.scatter(abs(r["validation_elo"] - int(T)), r["validation_sce"], label=r["method"])
        ax.set(xlabel=f"absolute rating error at {T}", ylabel="validation human log-loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"calibration_pareto_T{T}.png", dpi=140)
        plt.close(fig)


def run(args):
    run_dir, out = Path(args.run), Path(args.out)
    targets = list(range(args.start, args.stop + 1, args.step))
    config = {"version": VERSION, "targets": targets, "dial_grid": list(DIAL_GRID), "temp_grid": list(TEMP_GRID),
              "scan_main": SCAN_MAIN, "validation_openings": VALIDATION_OPENINGS, "humanlike_moves": HUMANLIKE_MOVES}
    part = run_dir / "calibration.partial.json"
    if part.exists():
        state = json.loads(part.read_text())
        if state["config"] != config:
            raise SystemExit(f"{part} belongs to a different configuration; use another --run directory")
    else:
        state = {"config": config, "targets": {}, "references": {}, "steps": 0, "games": 0}
    scan_games = (len(DIAL_GRID) + len(targets) * len(TEMP_GRID)) * (48 + 2 * SCAN_MAIN)
    validation_games = len(targets) * 4 * VALIDATION_OPENINGS * 4
    final_games = len(targets) * (len(pipeline.openings.load()) - VALIDATION_OPENINGS) * 4
    total_games = scan_games + validation_games + final_games
    if args.dry_run:
        print(json.dumps({"targets": targets, "scan_games": scan_games, "validation_games": validation_games,
                          "final_games": final_games, "total_games": total_games,
                          "human_fidelity_bands": [T for T in targets if has_fidelity(T)]}, indent=1))
        return
    total_steps = 1 + 3 * len(targets)
    ctx = pipeline.Ctx()

    def keep():
        _atomic(part, state)

    def advance(note, n=1):
        state["games"] = state.get("games", 0) + n
        progress.tick("calibration", state["games"], total_games, "games", note=note)

    def step(note):
        state["steps"] = int("dial_grid" in state) + sum(
            int("temp_grid" in st) + int("selected" in st) + int("final" in st) for st in state["targets"].values())
        progress.tick("calibration:phase", state["steps"], total_steps, "steps", force=True, note=note)
        keep()
        write_outputs(state, out)

    try:
        progress.tick("calibration", state.get("games", 0), total_games, "games", force=True,
                      note="starting full-dial calibration")
        if "dial_grid" not in state:
            played = ctx.adaptive(ctx.base, {**pipeline.POLICY, "self_elo": 0, "oppo_elo": 0}, DIAL_GRID,
                                  "calibration:scan", lambda d: f"Maia dial {d}", seed=11000, n_main=SCAN_MAIN,
                                  on_progress=lambda: advance("global dial scan"))
            state["dial_grid"] = [[d, ctx.units.lich(gauntlet.rate(played[d], ctx.elo))] for d in DIAL_GRID]
            for d in DIAL_GRID:
                pipeline.save_games(run_dir / "games" / f"scan_dial_{d}.json", played[d])
        if state["steps"] == 0:
            step("global dial scan complete")
        for T in targets:
            key = str(T)
            st = state["targets"].setdefault(key, {})
            if "temp_grid" not in st:
                variants = {t: {"temp": t} for t in TEMP_GRID}
                played = ctx.adaptive(ctx.base, {**pipeline.POLICY, "self_elo": T, "oppo_elo": T}, variants,
                                      "calibration:scan", lambda t: f"dial {T}, temperature {t}",
                                      seed=12000 + T, n_main=SCAN_MAIN,
                                      on_progress=lambda T=T: advance(f"temperature scan @ {T}"))
                st["temp_grid"] = [[t, ctx.units.lich(gauntlet.rate(played[t], ctx.elo))] for t in TEMP_GRID]
                st["candidates"] = candidates(T, state["dial_grid"], st["temp_grid"])
                for t in TEMP_GRID:
                    pipeline.save_games(run_dir / "games" / f"scan_temp_{T}_{t}.json", played[t])
                keep()
                step(f"temperature scan @ {T} complete")
            for i, row in enumerate(st["candidates"]):
                if "validation_elo" in row:
                    continue
                pol = {**pipeline.POLICY, "self_elo": row["dial"], "oppo_elo": row["dial"], "temp": row["temp"]}
                games = gauntlet.play(ctx.cfg, ctx.base, pol, ctx.units.closest(T, 3),
                                      ctx.suite[:VALIDATION_OPENINGS] * 2, seed=13000 + T,
                                      name="calibration:validate", elo=ctx.elo,
                                      label=f"{row['method']} @ {T}", cycle=True)
                advance(f"validation and selection @ {T}", len(games))
                row.update(_rated(ctx, games, T))
                row["validation_elo"] = row.pop("elo")
                row["validation_elo_ci"] = row.pop("elo_ci")
                row["validation_elo_ci_units"] = row.pop("elo_ci_units")
                row["validation_ladder"] = row.pop("ladder")
                if has_fidelity(T):
                    fid = _fid(ctx, T, row["dial"], row["temp"], final=False)
                    row["validation_top1"] = fid["top1"]
                    row["validation_sce"] = fid["sce"]
                    row["validation_fidelity"] = fid
                pipeline.save_games(run_dir / "games" / f"validation_{T}_{i}.json", games)
                keep()
            if "selected" not in st:
                best = choose(st["candidates"], T)
                st["selected"] = {k: best[k] for k in ("method", "dial", "temp", "reaches")}
                keep()
            step(f"validation and selection @ {T} complete")
            if "final" not in st:
                sel = st["selected"]
                pol = {**pipeline.POLICY, "self_elo": sel["dial"], "oppo_elo": sel["dial"], "temp": sel["temp"]}
                games = gauntlet.play(ctx.cfg, ctx.base, pol, ctx.units.closest(T, 3),
                                      ctx.suite[VALIDATION_OPENINGS:] * 2, seed=14000 + T,
                                      name="calibration:confirm", elo=ctx.elo,
                                      label=f"selected {sel['method']} @ {T}", cycle=True)
                advance(f"fresh-game confirmation @ {T}", len(games))
                final = _rated(ctx, games, T)
                if has_fidelity(T):
                    final["fidelity"] = _fid(ctx, T, sel["dial"], sel["temp"], final=True)
                    final["humanlike"] = evals.humanlike(evals.moves_of(games, HUMANLIKE_MOVES),
                                                         job="calibration:humanlike", what=f"selected {sel['method']} @ {T}")
                    final["reference"] = _reference(ctx, T, state, run_dir)
                st["final"] = final
                pipeline.save_games(run_dir / "games" / f"final_{T}.json", games)
                keep()
            step(f"fresh-game confirmation @ {T} complete")
        write_outputs(state, out)
        plot_outputs(state, out)
        progress.tick("calibration", total_games, total_games, "games", force=True, note="full-dial calibration complete")
    finally:
        arena.close_pool()


def main():
    ap = argparse.ArgumentParser(description="Calibrate Maia's dial and temperature across the full rating range")
    ap.add_argument("--run", default="runs/calibration", help="resumable games and partial state")
    ap.add_argument("--out", default="results/calibration", help="JSON, Markdown, map, and plots")
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--stop", type=int, default=2600)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true", help="print the workload without loading Maia or playing games")
    args = ap.parse_args()
    if args.start < 100 or args.stop < args.start or args.step <= 0:
        ap.error("require 100 <= start <= stop and a positive step")
    run(args)


if __name__ == "__main__":
    main()
