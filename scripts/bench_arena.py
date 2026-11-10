import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from levelrl import arena, ladder, pipeline

# Replays anchor gauntlets and prints how long they took, the load while they ran and the ladder Elo against the
# stored anchors. Run with LEVELRL_MOVETIME=0.05 to compare against a run made with the time-based rungs.


def main():
    ctx = pipeline.Ctx()
    old = {r["dial"]: r for r in json.loads((pipeline.RUN / "anchors.json").read_text())["dials"]}
    dials = [int(x) for x in sys.argv[1:]] or [1000, 1600]
    t0 = time.time()
    played = ctx.adaptive(ctx.base, {**pipeline.BOT, "self_elo": 0, "oppo_elo": 0}, dials, "bench", lambda d: f"maia3@{d}",
                          seed=1000, n_main=100)
    dt = time.time() - t0
    n = sum(len(g) for g in played.values())
    plies = sum(g["plies"] - len(g["opening"]) for gs in played.values() for g in gs)
    print(f"{n} games in {dt:.0f} s, {plies / 2 / dt:.0f} engine moves/s, load {os.getloadavg()[0]:.1f}, limit {ladder.LIMIT}")
    for d in dials:
        r = ctx.rated(played[d])
        print(f"dial {d}: ladder {r['ladder']:.0f} ± {1.96 * r['se']:.0f} (stored {old[d]['ladder']:.0f} ± "
              f"{1.96 * old[d]['se']:.0f})", flush=True)
    arena.close_pool()


if __name__ == "__main__":
    main()
