from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from levelrl import calibrate, data, evals, gauntlet, pipeline, progress

TARGETS = tuple(range(600, 2601, 100))
REQUIRED_ROWS = evals.BAND_N + evals.VAL_N + evals.FID_N


@lru_cache(maxsize=None)
def _band_audit(T: int, mtime: int):
    rows = data.load(f"fid_{T}")
    val = rows[evals.BAND_N:evals.BAND_N + evals.VAL_N]
    fid = rows[-evals.FID_N:]
    val_sites, fid_sites = {r["site"] for r in val}, {r["site"] for r in fid}
    val_months, fid_months = {r["month"] for r in val}, {r["month"] for r in fid}
    reasons = []
    if len(rows) < REQUIRED_ROWS:
        reasons.append(f"needs {REQUIRED_ROWS:,} rows")
    if val_sites & fid_sites:
        reasons.append("validation and final games overlap")
    if val_months & fid_months:
        reasons.append("validation and final months overlap")
    return {"target": T, "rows": len(rows), "months": sorted({r["month"] for r in rows}),
            "ready": not reasons, "reasons": reasons or ["ready"], "file": f"fid_{T}.jsonl.gz", "mtime": mtime}


def band_audit(T: int):
    path = data.ROOT / "positions" / f"fid_{T}.jsonl.gz"
    if not path.exists():
        return {"target": T, "rows": 0, "months": [], "ready": False, "reasons": ["missing band"], "file": path.name}
    return _band_audit(T, path.stat().st_mtime_ns)


def audit():
    rows = [band_audit(T) for T in TARGETS]
    return {"targets": rows, "ready": [r["target"] for r in rows if r["ready"]],
            "missing": [r["target"] for r in rows if not r["ready"]],
            "scan_cache": [p.parent.name for p in (data.ROOT / "scan").glob("*/done")],
            "required_rows_per_band": REQUIRED_ROWS}


def _map_exact(out: Path, T: int, selected: dict):
    path = out / "calibration_map.json"
    if not path.exists():
        return False
    entries = json.loads(path.read_text()).get("validated", [])
    row = next((r for r in entries if r["target"] == T), None)
    return bool(row and row.get("validated") and
                (row["method"], row["dial"], row["temp"]) == (selected["method"], selected["dial"], selected["temp"]))


def readiness(state, out: Path):
    rows = []
    for T in TARGETS:
        st = state.get("targets", {}).get(str(T), {})
        final = st.get("final", {})
        band = band_audit(T)
        strength = bool(final and abs(final["elo"] - T) <= calibrate.BAND)
        fidelity = bool(band["ready"] and final.get("fidelity") and final.get("humanlike") and final.get("reference"))
        exact = bool(st.get("selected") and _map_exact(out, T, st["selected"]))
        reasons = []
        if not final:
            reasons.append("no confirmed setting")
        elif not strength:
            reasons.append(f"strength error {final['elo'] - T:+.0f}")
        if not band["ready"]:
            reasons.extend(band["reasons"])
        elif not fidelity:
            reasons.append("final fidelity/reference analysis missing")
        if not exact:
            reasons.append("map is missing or differs from measured setting")
        rows.append({"target": T, "strength_ready": strength, "fidelity_ready": fidelity, "map_exact": exact,
                     "complete": strength and fidelity and exact, "reasons": reasons or ["ready"],
                     "selected": st.get("selected"), "elo": final.get("elo")})
    return {"complete": all(r["complete"] for r in rows), "complete_targets": sum(r["complete"] for r in rows),
            "total_targets": len(rows), "targets": rows}


def write_readiness(state, out: Path):
    report = readiness(state, out)
    calibrate._atomic(out / "coverage.json", report)
    lines = ["# Full-coverage readiness", "",
             f"**{report['complete_targets']}/{report['total_targets']} targets complete.** "
             "A target counts only when strength, human fidelity, and the exact measured map entry are all present.", "",
             "| target | strength | fidelity | exact map | status |", "|---|---|---|---|---|"]
    for r in report["targets"]:
        mark = lambda ok: "yes" if ok else "no"
        lines.append(f"| {r['target']} | {mark(r['strength_ready'])} | {mark(r['fidelity_ready'])} | "
                     f"{mark(r['map_exact'])} | {'; '.join(r['reasons'])} |")
    (out / "coverage.md").write_text("\n".join(lines) + "\n")
    return report


def build_data(args):
    report = audit()
    missing = [r["target"] for r in report["targets"] if not r["ready"]]
    plan = {"months": args.months, "bands": missing, "scan_cache": report["scan_cache"],
            "downloads_required": [m for m in args.months if m not in report["scan_cache"]]}
    print(json.dumps(plan, indent=1))
    if args.dry_run or not missing:
        return
    data.build(tuple(args.months), seed=args.seed, bands=tuple(missing))


def local_candidates(selected: dict, T: int):
    out = []
    for dd in (-40, 0, 40):
        for dt in (-0.04, 0.0, 0.04):
            d, t = max(100, selected["dial"] + dd), max(0.05, round(selected["temp"] + dt, 3))
            row = {"method": "repair", "dial": d, "temp": t, "reaches": True}
            if (d, t) != (selected["dial"], selected["temp"]):
                out.append(row)
    if T >= 2300:
        for t in (0.05, 0.1):
            if abs(selected["temp"] - t) > 1e-9:
                out.append({"method": "repair cold", "dial": selected["dial"], "temp": t, "reaches": True})
    unique = []
    for row in out:
        if (row["dial"], row["temp"]) not in {(r["dial"], r["temp"]) for r in unique}:
            unique.append(row)
    return unique


def _base_games(ctx, T, run_dir, on_progress=None):
    original = pipeline.RUN / "games" / f"base_{T}.json"
    if original.exists():
        return original
    path = Path(run_dir) / "games" / f"base_{T}.json"
    if not path.exists():
        games = gauntlet.play(ctx.cfg, ctx.base, {**pipeline.POLICY, "self_elo": T, "oppo_elo": T},
                              ctx.units.closest(T, 3), ctx.suite[calibrate.VALIDATION_OPENINGS:] * 2,
                              seed=15000 + T, name="calibration:base", elo=ctx.elo,
                              label=f"base model @ {T}", cycle=True)
        pipeline.save_games(path, games)
        if on_progress:
            on_progress(len(games))
    return path


def repair_remaining(state, targets, planned, run_dir):
    validation = final = reference = 0
    for T in targets:
        st = state["targets"][str(T)]
        rp = st.get("repair", {})
        done = sum("validation_elo" in r for r in rp.get("candidates", []))
        validation += (len(planned[T]) - done) * calibrate.VALIDATION_OPENINGS * 4
        n_final = (len(pipeline.openings.load()) - calibrate.VALIDATION_OPENINGS) * 4
        final += 0 if rp.get("final_complete") else n_final
        original = pipeline.RUN / "games" / f"base_{T}.json"
        generated = Path(run_dir) / "games" / f"base_{T}.json"
        if calibrate.has_fidelity(T) and str(T) not in state.get("references", {}) and not original.exists() and not generated.exists():
            reference += n_final
    return {"validation_games": validation, "final_games": final, "reference_games": reference,
            "total": validation + final + reference}


def repair(args):
    run_dir, out = Path(args.run), Path(args.out)
    state = json.loads((run_dir / "calibration.partial.json").read_text())
    targets = args.targets or [T for T in TARGETS if str(T) in state["targets"] and
                              abs(state["targets"][str(T)]["final"]["elo"] - T) > calibrate.BAND]
    def original(T):
        st = state["targets"][str(T)]
        history = st.get("repair", {}).get("history", [])
        return history[0]["selected"] if history else st["selected"]
    planned = {T: local_candidates(original(T), T) for T in targets}
    remaining = repair_remaining(state, targets, planned, run_dir)
    if args.dry_run:
        print(json.dumps({"targets": targets, "candidates": {T: len(c) for T, c in planned.items()},
                          "estimated_new_games": remaining["total"], **remaining}, indent=1))
        return
    ctx = pipeline.Ctx()
    total = state.get("games", 0) + remaining["total"]

    def advance(note, n):
        state["games"] = state.get("games", 0) + n
        progress.tick("calibration", state["games"], total, "games", note=note)

    try:
        for T in targets:
            st = state["targets"][str(T)]
            signature = [[r["dial"], r["temp"]] for r in planned[T]]
            rp = st.setdefault("repair", {"signature": signature, "candidates": [], "history": []})
            if rp["signature"] != signature:
                raise SystemExit(f"repair configuration changed for {T}; use a new --run directory")
            if not rp["history"]:
                rp["history"].append({"selected": st["selected"], "final": st["final"]})
            restored = []
            for row in planned[T]:
                old = next((r for r in rp["candidates"] if (r["dial"], r["temp"]) == (row["dial"], row["temp"])), None)
                if old is None:
                    old = next((r for r in st["candidates"] if (r["dial"], r["temp"]) == (row["dial"], row["temp"])), None)
                restored.append({**(old or {}), **row})
            rp["candidates"] = restored
            for i, row in enumerate(rp["candidates"]):
                if "validation_elo" in row:
                    continue
                pol = {**pipeline.POLICY, "self_elo": row["dial"], "oppo_elo": row["dial"], "temp": row["temp"]}
                games = gauntlet.play(ctx.cfg, ctx.base, pol, ctx.units.closest(T, 3),
                                      ctx.suite[:calibrate.VALIDATION_OPENINGS] * 2, seed=16000 + T,
                                      name="calibration:validate", elo=ctx.elo,
                                      label=f"repair {row['dial']}/{row['temp']} @ {T}", cycle=True)
                advance(f"local repair @ {T}", len(games))
                rated = calibrate._rated(ctx, games, T)
                row["validation_elo"] = rated["elo"]
                row["validation_elo_ci"] = rated["elo_ci"]
                row["validation_elo_ci_units"] = rated["elo_ci_units"]
                row["validation_ladder"] = rated["ladder"]
                if calibrate.has_fidelity(T):
                    fid = calibrate._fid(ctx, T, row["dial"], row["temp"], final=False)
                    row["validation_top1"] = fid["top1"]
                    row["validation_sce"] = fid["sce"]
                    row["validation_fidelity"] = fid
                pipeline.save_games(run_dir / "games" / f"repair_{T}_{i}.json", games)
                calibrate._atomic(run_dir / "calibration.partial.json", state)
            if rp.get("final_complete"):
                continue
            best = calibrate.choose(st["candidates"] + rp["candidates"], T)
            st["selected"] = {k: best[k] for k in ("method", "dial", "temp", "reaches")}
            pol = {**pipeline.POLICY, "self_elo": best["dial"], "oppo_elo": best["dial"], "temp": best["temp"]}
            games = gauntlet.play(ctx.cfg, ctx.base, pol, ctx.units.closest(T, 3),
                                  ctx.suite[calibrate.VALIDATION_OPENINGS:] * 2, seed=17000 + T,
                                  name="calibration:confirm", elo=ctx.elo, label=f"repair selected @ {T}", cycle=True)
            advance(f"fresh confirmation after repair @ {T}", len(games))
            final = calibrate._rated(ctx, games, T)
            if calibrate.has_fidelity(T):
                _base_games(ctx, T, run_dir, on_progress=lambda n: advance(f"base reference games @ {T}", n))
                final["fidelity"] = calibrate._fid(ctx, T, best["dial"], best["temp"], final=True)
                final["humanlike"] = evals.humanlike(evals.moves_of(games, calibrate.HUMANLIKE_MOVES),
                                                     job="calibration:humanlike", what=f"repair selected @ {T}")
                final["reference"] = calibrate._reference(ctx, T, state, run_dir)
            st["final"] = final
            rp["final_complete"] = True
            pipeline.save_games(run_dir / "games" / f"repair_final_{T}.json", games)
            calibrate._atomic(run_dir / "calibration.partial.json", state)
            calibrate.write_outputs(state, out)
        calibrate.plot_outputs(state, out)
        progress.tick("calibration", total, total, "games", force=True, note="targeted repairs complete")
    finally:
        calibrate.arena.close_pool()


def finalize(args):
    run_dir, out = Path(args.run), Path(args.out)
    state = json.loads((run_dir / "calibration.partial.json").read_text())
    targets = [T for T in TARGETS if str(T) in state["targets"] and calibrate.has_fidelity(T)
               and not all(state["targets"][str(T)]["final"].get(k) for k in ("fidelity", "humanlike", "reference"))]
    if args.dry_run:
        n = (len(pipeline.openings.load()) - calibrate.VALIDATION_OPENINGS) * 4
        reference_games = sum(n for T in targets if not (pipeline.RUN / "games" / f"base_{T}.json").exists()
                              and not (run_dir / "games" / f"base_{T}.json").exists())
        print(json.dumps({"targets": targets, "reference_games": reference_games}, indent=1))
        return
    ctx = pipeline.Ctx()
    try:
        for i, T in enumerate(targets, 1):
            st = state["targets"][str(T)]
            sel = st["selected"]
            path = run_dir / "games" / (f"repair_final_{T}.json" if st.get("repair", {}).get("final_complete")
                                       else f"final_{T}.json")
            games = json.loads(path.read_text())
            _base_games(ctx, T, run_dir)
            st["final"]["fidelity"] = calibrate._fid(ctx, T, sel["dial"], sel["temp"], final=True)
            st["final"]["humanlike"] = evals.humanlike(evals.moves_of(games, calibrate.HUMANLIKE_MOVES),
                                                       job="calibration:humanlike", what=f"final analysis @ {T}")
            st["final"]["reference"] = calibrate._reference(ctx, T, state, run_dir)
            calibrate._atomic(run_dir / "calibration.partial.json", state)
            calibrate.write_outputs(state, out)
            progress.tick("calibration", i, len(targets), "targets", force=True, note=f"final fidelity analysis @ {T}")
        calibrate.plot_outputs(state, out)
    finally:
        calibrate.arena.close_pool()


def main():
    ap = argparse.ArgumentParser(description="Audit and complete every Maia calibration target")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("audit", "build-data", "repair", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        if name == "build-data":
            p.add_argument("--months", nargs="+", default=list(data.MONTHS))
            p.add_argument("--seed", type=int, default=0)
        if name in ("repair", "finalize"):
            p.add_argument("--run", default="runs/calibration")
            p.add_argument("--out", default="results/calibration")
        if name == "repair":
            p.add_argument("--targets", nargs="+", type=int)
    args = ap.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(), indent=1))
    elif args.command == "build-data":
        build_data(args)
    elif args.command == "repair":
        repair(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
