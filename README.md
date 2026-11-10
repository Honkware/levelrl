# LevelRL

GRPO post-training of [Maia-3-79M](https://huggingface.co/UofTCSSLab/Maia3-79M) so that it plays at the Lichess
blitz rating it is asked for, while still choosing the moves people at that rating choose. This is independent
work and is not affiliated with or endorsed by CSSLab.

Status: one full run is finished and written up in `results/run1/summary.md`. A second run with the changes it
called for (below) is in progress, so nothing here is a final result yet.

## Why

Maia-3 takes a rating dial, and the dial is off. Measured against the `maia3-79m_*` bots on Lichess, a dial of 600
plays around 890 blitz and the scale flattens out near 2200. Weakened engines hit a rating but do not lose the
way people do. The question here is how much rating accuracy can be bought, and what it costs in move-matching
against real games.

## How it works

- `levelrl/data.py` streams post-cutoff Lichess dumps (the model saw January 2023 to July 2025, so everything
  here is from 2025-09 or later) into calibration and per-band fidelity positions. A game from inside the
  training window stops the build.
- `levelrl/ladder.py` is a ladder of Stockfish 19 rungs at 30,000 nodes a move, fitted with Bradley-Terry.
  `runs/elo_calibration.json` has the ratings.
- `levelrl/anchors.py` and `levelrl/units.py` turn ladder Elo into Lichess blitz Elo. The base model plays the
  ladder the way the Lichess bots sample (history on, temperature 1.1, top-p 0.94, inferred from their games), and
  each dial's ladder rating is fitted against that bot's Lichess rating. Lichess is only ever read, with no
  account and no games played.
- `levelrl/train.py` is the trainer. Groups of 8 games share an opening and an opponent. The loss is
  `w_out * PPO(pull * outcome) + w_fid * 4 * KL(policy || base model at the target)`. `pull` runs from +1 (150 or
  more Elo below the target, winning is rewarded) to -1 (150 or more above, losing is rewarded) and is zero on
  the target. It follows a rating of the last 256 training games, refreshed every update.
- `levelrl/evals.py` holds what never feeds a reward: move-match on held-out human positions, blunder types and
  per-phase error rates judged by Stockfish, opening-book overlap and repetition rate.
- `levelrl/pipeline.py` runs all of it: snapshot, anchors, Maia-1 cross-check, base curve, smoke run, a sweep of
  five reward weightings per target, and the report.

## Running it

```
pip install -r requirements.txt
(cd native && cargo build --release)      # optional Rust board/token/legal-mask acceleration
scripts/run_pipeline.sh                  # tmux session "pipeline", resumes where it stopped
python -m pytest -q
```

It needs `stockfish` (19) on the path, a CUDA GPU, and the position files built by `levelrl/data.py`. The Maia-1
cross-check is skipped unless lc0 and the Maia-1 nets are present (`scripts/setup_crosscheck.sh`). If the native
library is not built, the exact Python feature implementation is used instead.

## What the first run taught

Run 1 used 50 ms rungs, a sampled "judge" reward (the base model's log-prob of the move played) for fidelity, and
set the pull from a gauntlet every 25 updates. Its numbers are in `results/run1/`.

- The pull worked: at target 1100, where the base model plays 1257, three of five arms finished inside 1100 ± 50.
- It did not hold the band. Elo swung 100 to 250 points between evals (1720, 1470, 1602, 1480 on one arm), and
  the reported checkpoint was the best of four. The pull now follows the training games themselves.
- The judge reward was gamed. It is only the cross-entropy half of a KL, so the policy raised it by sharpening:
  entropy fell, strength rose, and move-match on held-out humans dropped 0.4 to 2.4 points on every arm,
  including the fidelity-only one. Seven of the eight hack-catalog entries are this. Fidelity is now the whole
  KL to the base model, as an exact loss term.
- The 1500 target tested nothing, since the base model already played 1494 there. Targets are now the dial
  that plays most above its setting and the one most below.
- Final intervals were about ± 50 Elo, as wide as the band. Confirmation gauntlets are now 800 games.
- Rung strength depended on CPU load. Rungs now search a fixed 30,000 nodes, the median they reached in 50 ms
  with every worker busy.

## Sharp edges

- The ladder is compressed (in run 1, Stockfish's nominal 1320 to 2500 spanned about 730 ladder points), so the
  unit conversion fits a slope as well as an offset. Elo intervals in the results carry the gauntlet's error
  through that fit and leave out the error of the fit itself. Run 1's fit missed single anchors by up to 96
  points below dial 2200, and predicted the Maia-1 bots 37, 187 and 121 points too high.
- The bots above dial 2200 have few rated games and are treated as plus or minus 150.
- Fidelity is measured against the base model, not against people. The held-out human move-match and the hack
  catalog are the check on that.
- `KL_COEF = 4` and `PULL = 150` in `levelrl/train.py` are first guesses that no run has tuned.
- The human-likeness table has no intervals yet.

## Credits and licence

Maia-3 is by the CSSLab at the University of Toronto ([arXiv:2605.19091](https://arxiv.org/abs/2605.19091), code
at [CSSLab/maia3](https://github.com/CSSLab/maia3), AGPL-3.0). This repository is
AGPL-3.0.
