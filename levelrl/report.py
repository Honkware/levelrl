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
    baselines = rd("baselines") if (run / "baselines.json").exists() else {}
    arms = {n: rd(f"arm_{n}") for n in sweep["arms"]}
    paretos, catalog = {}, []
    for name, a in arms.items():
        T, f = a["T"], a["final"]
        fid = f["fidelity"] or {}
        curve = [{"tag": e["tag"], "update": e["update"], "elo": e["elo"], "elo_ci": e["elo_ci"],
                  "top1": e["probe"].get(f"band_{T}", e["probe"]["calib"])["top1"],
                  "sce": e["probe"].get(f"band_{T}", e["probe"]["calib"])["sce"]} for e in a["evals"] if "elo" in e]
        paretos.setdefault(str(T), {})[name] = {
            "goal": a["goal"], "start": a["start"], "frac": a["frac"], "elo": f["elo"], "elo_ci": f["elo_ci"],
            "elo_ci_units": f.get("elo_ci_units"), "ladder": f["ladder"]["ladder"], "distance_to_goal": f["elo"] - a["goal"],
            "fidelity_top1": fid.get("top1"), "fidelity_sce": fid.get("sce"),
            "fidelity_top1_vs_base_ci": fid.get("top1_vs_base_ci"), "fidelity_sce_vs_base_ci": fid.get("sce_vs_base_ci"),
            "keep_point_fidelity_drop": fid.get("top1_vs_base_ci") is not None and fid["top1_vs_base_ci"][1] < -FID_DROP_PP,
            "kl_base": next((x["kl_base"] for x in reversed(a.get("train_log", [])) if "kl_base" in x), None),
            "weight": a.get("weight"), "updates": a.get("updates"), "games": a.get("games"),
            "stopped_early": a.get("stopped_early"), "held": a.get("held", {}), "bias": a.get("bias"), "band_pick": a.get("band_pick"),
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
    man = rd("manifest") if (run / "manifest.json").exists() else {}
    (res / "manifest.json").write_text(json.dumps(man, indent=1))
    (res / "baselines.json").write_text(json.dumps(baselines, indent=1))
    summary(res, anchors, base, cross, paretos, catalog, sweep, man, baselines)
    try:
        plots(res, anchors, base, paretos)
    except Exception as e:  # the tables are the result, a plotting failure must not cost them
        print("plots failed:", repr(e), flush=True)
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
                        fmt="o", capsize=3, label=f"goal {a['goal']:.0f}")
            ax.axvline(a["goal"], color="k", ls=":", lw=0.5)
        ax.set_xlabel("Lichess blitz Elo (last checkpoint)")
        ax.set_ylabel(f"top-1 on held-out humans at {T} (%)")
        ax.set_title(f"Dial {T}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(res / f"pareto_T{T}.png", dpi=130)
    plt.close("all")


def _hl(name: str, h: dict) -> str:
    ph = h["by_phase"]
    pm = lambda p, n: f"{100 * p:.1f} ± {196 * (p * (1 - p) / max(1, n)) ** 0.5:.1f}"
    cell = lambda k: pm(ph[k]["mistake_rate"], ph[k]["n"]) if k in ph else "n/a"
    return (f"| {name} | {h['n']} | {pm(h['blunder_rate'], h['n'])} | {100 * h['blunder_types']['piece drop']:.0f} | "
            f"{cell('opening')} | {cell('middlegame')} | {cell('endgame')} |")


def _arm(a: dict) -> str:
    return f"{a['frac']:.0%} of the way, {a['goal']:.0f}"


def summary(res, anchors, base, cross, paretos, catalog, sweep, man=None, baselines=None):
    f = anchors["fit"][anchors["fit"]["use"]]
    partial = [f"> **Partial run:** {sweep['partial']}", ""] if sweep.get("partial") else []
    L = partial + ["# LevelRL results", "", f"Code: commit {(man or {}).get('commit') or 'not recorded'}.", "",
         f"Units: Lichess blitz = {f['a']:.0f} + {f['b']:.3f} x ladder ({anchors['fit']['use']} fit, "
         f"slope 95% CI {f['b_ci'][0]:.3f} to {f['b_ci'][1]:.3f}), snapshot {anchors['snapshot']}. "
         "Elo intervals marked \"in run\" are the gauntlet's bootstrap interval carried through this fit, which is "
         "what compares arms with each other. Those marked \"with units\" add the fit's own uncertainty, which is "
         "what an absolute Lichess number carries. The fit is only tested on Maia-3 players: see the cross-check.", "",
         "## Anchors", "", "| dial | ladder | Lichess | fit residual |", "|---|---|---|---|"]
    L += [f"| {r['dial']} | {r['ladder']:.0f} | {r['lichess']} | {x:+.0f} |" for r, x in zip(anchors["dials"], f["resid"])]
    L += ["", "## Base conditioning curve", "", "| dial | ladder | Lichess Elo | 95% CI in run | with units | off by | games |",
          "|---|---|---|---|---|---|---|"]
    cu = lambda r: (f"{r['lichess_ci_units'][0]:.0f} to {r['lichess_ci_units'][1]:.0f}" if r.get("lichess_ci_units") else "n/a")
    L += [f"| {r['dial']} | {r['ladder']:.0f} | {r['lichess']:.0f} | {r['lichess_ci'][0]:.0f} to {r['lichess_ci'][1]:.0f} | "
          f"{cu(r)} | {r['lichess'] - r['dial']:+.0f} | {r['games']} |" for r in base["curve"]]
    L += ["", f"Targets: {base['targets']}.", ""]
    if cross.get("bots"):
        L += ["## Maia-1 cross-check", "", "| bot | predicted | Lichess | error |", "|---|---|---|---|"]
        L += [f"| {b['bot']} | {b['predicted_lichess']:.0f} | {b['lichess']} | {b['predicted_lichess'] - b['lichess']:+.0f} |"
              for b in cross["bots"]]
        L.append("")
    ref = sweep.get("humanlike_reference", {})
    for T, arms in paretos.items():
        start = next(iter(arms.values()))["start"]
        L += [f"## Dial {T}", "", f"The base model plays {start:.0f} here. Each arm's goal is a share of the way to {T}, "
              "and the checkpoint reported is the last one, with no selection.", "",
              "Where each arm landed:", "",
              "| goal | ladder | Elo | 95% CI in run | with units | vs goal | in band, last third | rating sd, last third | outcome weight at the end | training games |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for a in arms.values():
            h, cu = a.get("held") or {}, a.get("elo_ci_units")
            L.append(f"| {_arm(a)} | {a['ladder']:.0f} | {a['elo']:.0f} | {a['elo_ci'][0]:.0f} to {a['elo_ci'][1]:.0f} | "
                     + (f"{cu[0]:.0f} to {cu[1]:.0f}" if cu else "n/a") + f" | {a['distance_to_goal']:+.0f} | "
                     + (f"{h['in_band']:.0%} | {h['sd']:.0f}" if h else "n/a | n/a")
                     + (f" | {a['weight']:+.2f} |" if a.get("weight") is not None else " | n/a |")
                     + (f" {a['games']:,} ({'stopped early' if a.get('stopped_early') else 'the cap'}) |" if a.get("games") else " n/a |"))
        bl = ((baselines or {}).get(T) or (baselines or {}).get(str(T)) or {}).get("methods", [])
        if bl:
            L += ["", "What the frozen model does with no training. Each is set to play this dial's Elo and judged on "
                  "the same 10,000 human positions at this band. An arm has to beat these to be worth training:", "",
                  "| method | dial | temperature | Elo | 95% CI in run | vs dial | top-1 | vs dial as asked (pp, 95% CI) | log-loss vs it (nats, 95% CI) |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for b in bl:
                fd = b["fidelity"]
                L.append(f"| {b['method']}{'' if b['reaches'] else ' (grid never reaches the dial)'} | {b['dial']} | {b['temp']} | "
                         f"{b['elo']:.0f} | {b['elo_ci'][0]:.0f} to {b['elo_ci'][1]:.0f} | {b['elo'] - int(T):+.0f} | {fd['top1']:.1f} | "
                         f"{fd['top1_vs_base_ci'][0]:+.1f} to {fd['top1_vs_base_ci'][1]:+.1f} | "
                         f"{fd['sce_vs_base_ci'][0]:+.3f} to {fd['sce_vs_base_ci'][1]:+.3f} |")
        L += ["", "What it cost in human-likeness, on 10,000 held-out human positions at this band:", "",
              "| goal | Elo moved | top-1 | vs base (pp, 95% CI) | log-loss vs base (nats, 95% CI) | KL to base |",
              "|---|---|---|---|---|---|"]
        for a in arms.values():
            ci, sc = a["fidelity_top1_vs_base_ci"], a["fidelity_sce_vs_base_ci"]
            top1 = f"{a['fidelity_top1']:.1f}" if a["fidelity_top1"] is not None else "n/a"
            L.append(f"| {_arm(a)} | {a['elo'] - start:+.0f} | {top1} | "
                     + (f"{ci[0]:+.1f} to {ci[1]:+.1f}" if ci else "n/a") + " | "
                     + (f"{sc[0]:+.3f} to {sc[1]:+.3f}" if sc else "n/a") + " | "
                     + (f"{a['kl_base']:.3f} |" if a.get("kl_base") is not None else "n/a |"))
        L += ["", "Band-aware pick: of the evals that measured within 50 Elo of the goal, the checkpoint with the best "
              "move-match, with its Elo from a fresh gauntlet. This one is selected, so read it beside the last "
              "checkpoint above, not in place of it.", "",
              "| goal | checkpoint | Elo | 95% CI in run | vs goal | top-1 | vs base (pp, 95% CI) |", "|---|---|---|---|---|---|---|"]
        for a in arms.values():
            b = a.get("band_pick")
            if not b:
                L.append(f"| {_arm(a)} | none measured in the band | | | | | |")
                continue
            fd = b.get("fidelity") or {}
            ci = fd.get("top1_vs_base_ci")
            L.append(f"| {_arm(a)} | {b['tag']} | {b['elo']:.0f} | {b['elo_ci'][0]:.0f} to {b['elo_ci'][1]:.0f} | "
                     f"{b['elo'] - a['goal']:+.0f} | " + (f"{fd['top1']:.1f}" if fd.get("top1") is not None else "n/a") + " | "
                     + (f"{ci[0]:+.1f} to {ci[1]:+.1f} |" if ci else "n/a |"))
        L += ["", "Elo during training, at each eval:", "", "| goal | " + " | ".join(
            c["tag"] for c in max((a["curve"] for a in arms.values()), key=len)) + " |",
              "|---|" + "---|" * max(len(a["curve"]) for a in arms.values())]
        L += [f"| {_arm(a)} | " + " | ".join(f"{c['elo']:.0f}" for c in a["curve"]) + " |" for a in arms.values()]
        L += ["", "Conditioning sensitivity of each final checkpoint (Elo when the dial is moved):", "",
              "| goal | dial-200 | dial | dial+200 | span |", "|---|---|---|---|---|"]
        for a in arms.values():
            e = [v["elo"] for _, v in sorted(a["sensitivity"].items(), key=lambda kv: int(kv[0]))]
            L.append(f"| {_arm(a)} | " + " | ".join(f"{x:.0f}" for x in e) + f" | {e[-1] - e[0]:+.0f} |")
        L += ["", "Human-likeness, outside the reward (percent of moves ± 95%, Stockfish at 100k nodes):", "",
              "| player | moves | blunders | of which piece drops | opening mistakes | middlegame | endgame |",
              "|---|---|---|---|---|---|---|"]
        r = ref.get(T) or ref.get(int(T)) or {}
        L += [_hl(f"humans at {T}", r["human"])] if "human" in r else []
        L += [_hl("base model", r["base"])] if "base" in r else []
        L += [_hl(f"no training: {b['method']}", b["humanlike"]) for b in bl if b["method"] != "dial as asked"]
        L += [_hl(_arm(a), a["humanlike"]) for a in arms.values()]
        L.append("")
    L += ["## Hack catalog", "", f"{len(catalog)} entries, in hack_catalog.json with example PGNs.", ""]
    L += [f"- {c['arm']}: {c['kind']}" + (f" at eval {c['eval']}" if "eval" in c else f", {c['from']:.3f} to {c['to']:.3f}")
          for c in catalog]
    (res / "summary.md").write_text("\n".join(L) + "\n")
