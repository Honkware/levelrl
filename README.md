# LevelRL

GRPO post-training of [Maia-3-79M](https://huggingface.co/UofTCSSLab/Maia3-79M) so that it plays at the Lichess
blitz rating it is asked for, while still choosing the moves people at that rating choose. This is independent
work and is not affiliated with or endorsed by CSSLab.

Status: the pipeline is running. The trained arms and their Pareto curves are not in yet, so `results/` only has
the gates and the ladder so far.

## Why

Maia-3 takes a rating dial, and the dial is off. Measured against the `maia3-79m_*` bots on Lichess, a dial of 600
plays around 890 blitz and the scale flattens out near 2200. Weakened engines hit a rating but do not lose the
way people do. The question here is how much rating accuracy can be bought, and what it costs in move-matching
against real games.

## How it works

- `levelrl/data.py` streams post-cutoff Lichess dumps (the model saw January 2023 to July 2025, so everything
  here is from 2025-09 or later) into calibration and per-band fidelity positions. A game from inside the
  training window stops the build.
- `levelrl/ladder.py` is a ladder of Stockfish 19 rungs at 50 ms a move, fitted with Bradley-Terry over 6,400
  games. `results/elo_calibration.json` has the ratings.
- `levelrl/anchors.py` and `levelrl/units.py` turn ladder Elo into Lichess blitz Elo. The base model plays the
  ladder the way the Lichess bots sample (history on, temperature 1.1, top-p 0.94, inferred from their games), and
  each dial's ladder rating is fitted against that bot's Lichess rating. Lichess is only ever read, with no
  account and no games played.
- `levelrl/train.py` is the trainer. Groups of 8 games share an opening and an opponent. The reward is
  `w_out * pull * outcome + w_fid * judge`, where the judge is the frozen base model's mean log-prob of the moves
  played at the target band. `pull` runs from +1 (150 or more Elo below the target, winning is rewarded) to -1
  (150 or more above, losing is rewarded) and is zero on the target. Outcome and judge are each standardised
  within a group before the weights apply, so 0.5/0.5 means an even split.
- `levelrl/evals.py` holds what never feeds a reward: move-match on held-out human positions, blunder types and
  per-phase error rates judged by Stockfish, opening-book overlap and repetition rate.
- `levelrl/pipeline.py` runs all of it: snapshot, anchors, Maia-1 cross-check, base curve, smoke run, a sweep of
  five reward weightings per target, and the report.

## Running it

```
pip install -r requirements.txt
scripts/run_pipeline.sh                  # tmux session "pipeline", resumes where it stopped
pytest -q
```

It needs `stockfish` (19) on the path, a CUDA GPU, and the position files built by `levelrl/data.py`. The Maia-1
cross-check is skipped unless lc0 and the Maia-1 nets are present (`scripts/setup_crosscheck.sh`).

## Sharp edges

- Ladder ratings depend on the machine. Rungs think for a fixed 50 ms, so a different CPU, or the same CPU under
  load, gives a different ladder. Rerun the calibration on new hardware and do not run anything heavy beside it.
- The ladder is compressed (Stockfish's nominal 1320 to 2500 spans about 730 ladder points), so the unit
  conversion fits a slope as well as an offset. Elo intervals in the results carry the gauntlet's error through
  that fit and leave out the error of the fit itself.
- The bots above dial 2200 have few rated games and are treated as plus or minus 150.
- The judge is the base model, so the policy can learn to please it. The held-out human move-match and the hack
  catalog are the check on that.

## Credits and licence

Maia-3 is by the CSSLab at the University of Toronto ([arXiv:2605.19091](https://arxiv.org/abs/2605.19091), code
at [CSSLab/maia3](https://github.com/CSSLab/maia3), AGPL-3.0). This repository is
AGPL-3.0.
