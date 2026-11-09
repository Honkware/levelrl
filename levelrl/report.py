from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.pgn

# Turns runs/pipeline into results/: paretos.json, the hack catalog with example games, the base curve, the
# sensitivity and human-likeness tables, plots, and a summary. Every number here is read from a run file.

FID_DROP_PP = 2.0


def _pgn(g: dict, me: str) -> str:
    b = chess.Board()
    for m in g["moves"]:
        b.push_uci(m)
    game = chess.pgn.Game.from_board(b)
    game.headers.update({"Event": "LevelRL", "White": me if g["white"] else g["opp"],
                         "Black": g["opp"] if g["white"] else me, "Termination": g.get("end", "")})
    return str(game)


def write(run: Path, res: Path) -> dict:
    res.mkdir(parents=True, exist_ok=True)
    rd = lambda n: json.loads((run / f"{n}.json").read_text())
    anchors, base, sweep = rd("anchors"), rd("base"), rd("sweep")
    cross = rd("crosscheck") if (run / "crosscheck.json").exists() else {}
    arms = {n: rd(f"arm_{n}") for n in sweep["arms"]}
    paretos, catalog = {}, []
    for name, a in arms.items():
        T, f = a["T"], a["final"]
        fid = f["fidelity"] or {}
        curve = [{"tag": e["tag"], "update": e["update"], "elo": e["elo"], "elo_ci": e["elo_ci"],
                  "top1": e["probe"].get(f"band_{T}", e["probe"]["calib"])["top1"],
                  "sce": e["probe"].get(f"band_{T}", e["probe"]["calib"])["sce"]} for e in a["evals"] if "elo" in e]
        paretos.setdefault(str(T), {})[name] = {
            "w_out": a["w_out"], "w_fid": a["w_fid"], "final_tag": f["tag"], "elo": f["elo"], "elo_ci": f["elo_ci"],
            "distance_to_T": f["elo"] - T, "fidelity_top1": fid.get("top1"), "fidelity_sce": fid.get("sce"),
            "fidelity_top1_vs_base_ci": fid.get("top1_vs_base_ci"), "fidelity_sce_vs_base_ci": fid.get("sce_vs_base_ci"),
            "keep_point_fidelity_drop": fid.get("top1_vs_base_ci") is not None and fid["top1_vs_base_ci"][1] < -FID_DROP_PP,
            "curve": curve, "sensitivity": a["sensitivity"], "humanlike": a["humanlike"], "recoveries": a["recoveries"]}
        # Hack catalog: per-arm drift signals, and every eval where the judge rose while human fidelity fell.
        for e in a["evals"]:
            if e.get("judge_gap"):
                d = run.parent / "arms" / name / f"eval_{e['tag']}.games.json"
                ex = json.loads(d.read_text())[:3] if d.exists() else []
                catalog.append({"arm": name, "eval": e["tag"], "kind": "judge gap", **e["judge_gap"],
                                "entropy": e["entropy"], "kl_base": e["kl_base"], "threefold": e["threefold"],
                                "book": e["book"], "examples": [_pgn(g, f"policy@{T}") for g in ex]})
        first, last = next((e for e in a["evals"] if "elo" in e), None), [e for e in a["evals"] if "elo" in e][-1:]
        if first and last:
            l = last[0]
            for k, rise in (("threefold", 0.05), ("kl_base", 0.5)):
                if l.get(k) is not None and first.get(k) is not None and l[k] - first[k] > rise:
                    d = run.parent / "arms" / name / f"eval_{l['tag']}.games.json"
                    ex = [g for g in json.loads(d.read_text()) if k != "threefold" or g["end"] == "threefold repetition"][:3]
                    catalog.append({"arm": name, "kind": f"{k} drift", "from": first[k], "to": l[k],
                                    "examples": [_pgn(g, f"policy@{T}") for g in ex]})
    (res / "paretos.json").write_text(json.dumps(paretos, indent=1))
    (res / "hack_catalog.json").write_text(json.dumps(catalog, indent=1))
    (res / "base_curve.json").write_text(json.dumps(base, indent=1))
    (res / "anchors.json").write_text(json.dumps(anchors, indent=1))
    (res / "crosscheck.json").write_text(json.dumps(cross, indent=1))
    (res / "humanlike_reference.json").write_text(json.dumps(sweep["humanlike_reference"], indent=1))
    plots(res, anchors, base, paretos)
    summary(res, anchors, base, cross, paretos, catalog, sweep)
    return {"arms": len(arms), "catalog": len(catalog)}


def plots(res, anchors, base, paretos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    c = base["curve"]
    ax.errorbar([r["dial"] for r in c], [r["lichess"] for r in c],
                yerr=[[r["lichess"] - r["lichess_ci"][0] for r in c], [r["lichess_ci"][1] - r["lichess"] for r in c]],
                fmt="o-", label="Maia-3-79M, plain sampling")
    d = anchors["dials"]
    ax.plot([r["dial"] for r in d], [r["lichess"] for r in d], "s", alpha=.6, label="maia3-79m bots on Lichess")
    ax.plot([600, 2600], [600, 2600], "k:", lw=1, label="dial = rating")
    ax.set_xlabel("self_elo dial")
    ax.set_ylabel("Lichess blitz Elo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(res / "base_curve.png", dpi=130)
    for T, arms in paretos.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, a in arms.items():
            if a["fidelity_top1"] is None:
                continue
            ax.errorbar(a["elo"], a["fidelity_top1"], xerr=[[a["elo"] - a["elo_ci"][0]], [a["elo_ci"][1] - a["elo"]]],
                        fmt="o", capsize=3, label=f"{a['w_out']}/{a['w_fid']}")
        ax.axvline(int(T), color="k", ls=":", lw=1)
        ax.set_xlabel("Lichess blitz Elo (final checkpoint)")
        ax.set_ylabel(f"top-1 on held-out humans at {T} (%)")
        ax.set_title(f"T = {T}, arms w_out/w_fid")
        ax.legend()
        fig.tight_layout()
        fig.savefig(res / f"pareto_T{T}.png", dpi=130)
    plt.close("all")


def summary(res, anchors, base, cross, paretos, catalog, sweep):
    f = anchors["fit"][anchors["fit"]["use"]]
    L = [f"# LevelRL results", "",
         f"Units: Lichess blitz = {f['a']:.0f} + {f['b']:.3f} x ladder ({anchors['fit']['use']} fit, "
         f"slope 95% CI {f['b_ci'][0]:.3f} to {f['b_ci'][1]:.3f}), snapshot {anchors['snapshot']}.", "",
         "## Base conditioning curve", "", "| dial | Lichess Elo | 95% CI | games |", "|---|---|---|---|"]
    L += [f"| {r['dial']} | {r['lichess']:.0f} | {r['lichess_ci'][0]:.0f} to {r['lichess_ci'][1]:.0f} | {r['games']} |"
          for r in base["curve"]]
    L += ["", f"Targets: {base['targets']}.", ""]
    if cross.get("bots"):
        L += ["## Maia-1 cross-check", "", "| bot | predicted | Lichess |", "|---|---|---|"]
        L += [f"| {b['bot']} | {b['predicted_lichess']:.0f} | {b['lichess']} |" for b in cross["bots"]]
        L.append("")
    for T, arms in paretos.items():
        L += [f"## T = {T}", "", "| arm w_out/w_fid | Elo | 95% CI | vs T | fidelity top-1 | vs base (pp, 95% CI) |",
              "|---|---|---|---|---|---|"]
        for a in arms.values():
            ci = a["fidelity_top1_vs_base_ci"]
            L.append(f"| {a['w_out']}/{a['w_fid']} | {a['elo']:.0f} | {a['elo_ci'][0]:.0f} to {a['elo_ci'][1]:.0f} | "
                     f"{a['distance_to_T']:+.0f} | {a['fidelity_top1'] or float('nan'):.1f} | "
                     + (f"{ci[0]:+.1f} to {ci[1]:+.1f} |" if ci else "n/a |"))
        L.append("")
    L += [f"## Hack catalog", "", f"{len(catalog)} entries, in hack_catalog.json with example PGNs.", ""]
    (res / "summary.md").write_text("\n".join(L))
