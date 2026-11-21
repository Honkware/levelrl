# LevelRL

[Maia-3](https://github.com/CSSLab/maia3) is a chess model from the CSSLab that takes a rating as input and plays like a human at that rating. The rating you give it and the strength you get are different, though. The Maia-3 bots on Lichess that are set to 600, 800 and 1000 have blitz ratings of 890, 1111 and 1289.

This repo measures that gap across the whole range and gives you the settings that close it. For each target rating from 600 to 2600 it has a dial value and a sampling temperature, found by search and then confirmed over 760 games on fresh openings. With those settings Maia-3 plays within 50 Elo of the rating you ask for at 16 of the 21 targets, with no training.

| asked for | dial | temperature | plays at | off by |
|---|---|---|---|---|
| 600 | 600 | 1.232 | 596 | -4 |
| 700 | 610 | 1.087 | 748 | +48 |
| 800 | 650 | 1.04 | 852 | +52 |
| 900 | 840 | 1.071 | 991 | +91 |
| 1000 | 740 | 1 | 987 | -13 |
| 1100 | 1100 | 1.297 | 1061 | -39 |
| 1200 | 990 | 1 | 1209 | +9 |
| 1300 | 1300 | 1.177 | 1282 | -18 |
| 1400 | 1250 | 1 | 1382 | -18 |
| 1500 | 1520 | 1 | 1532 | +32 |
| 1600 | 1600 | 1 | 1557 | -43 |
| 1700 | 1700 | 0.996 | 1666 | -34 |
| 1800 | 1910 | 1 | 1767 | -33 |
| 1900 | 2010 | 0.895 | 1914 | +14 |
| 2000 | 2130 | 0.876 | 1973 | -27 |
| 2100 | 2600 | 1 | 2058 | -42 |
| 2200 | 2400 | 0.763 | 2157 | -43 |
| 2300 | 2300 | 0.344 | 2347 | +47 |
| 2400 | 2460 | 0.51 | 2322 | -78 |
| 2500 | 2500 | 0.1 | 2415 | -85 |
| 2600 | 2600 | 0.1 | 2443 | -157 |

The five targets outside 50 Elo are 800, 900 and everything from 2400 up. The top of the range can't be fixed with settings, because the model's own ceiling is about 2440.

## Using it

The table ships with the `maia-calibrate` package, which has no dependencies and needs no GPU.

```sh
pip install maia-calibrate
```

```python
from maia_calibrate import maia3_79m

s = maia3_79m().lookup(1500).setting
print(s.dial, s.temp)   # 1520 1.0
```

Pass `s.dial` to Maia-3 as the rating for both players and sample moves at `s.temp`. A rating between two measured targets gets the nearest measured setting and is marked as not validated.

## Calibration compared with RL fine-tuning

I started this project to do the same job with RL, by fine-tuning Maia-3 with GRPO so that it holds a target rating while still choosing human moves. That works too, and the comparison at a target of 1100 is below. All three were measured in the same run over 800 games each.

| player | Elo | off target |
|---|---|---|
| dial 1100 as asked | 1305 | +205 |
| temperature 1.25, no training | 1078 | -22 |
| GRPO, last checkpoint | 1132 | +32 |

Training took about 1,500 to 3,500 games per arm and gives one checkpoint for one rating. In the second run it also lost between 0.8 and 5 points of human move-match. Calibration reaches the same ratings from the released weights, so that is what I'd use.

## How the ratings are measured

The model plays a gauntlet against Stockfish 19 at fixed node counts. The result is converted to a Lichess blitz rating with a fit against the real ratings of the Maia-3 bots. Human-likeness is top-1 agreement with the move a person played, over 10,000 positions from Lichess games at that rating. The tables for every run are in `results/`.

## Rerunning it

Training and the sweep need a CUDA GPU (I used one RTX 4090), Stockfish 19 on the path and a month of games from database.lichess.org.

```
pip install -r requirements.txt
(cd native && cargo build --release)
python -m levelrl.data --scan 2026-03
scripts/run_pipeline.sh
python -m pytest -q
```

## What's in the repo

- `levelrl/` has the experiment: the data build, the Stockfish ladder, the GRPO trainer and the calibration sweep.
- `maia_calibrate/` looks up the measured dial and temperature for a rating.
- `chess_eval/` rates any chess policy against fixed opponents and scores it on human positions.
- `native/` is a Rust crate with the board encoding and a PGN scanner.

## Licence and credits

Maia-3 is by the CSSLab at the University of Toronto and is AGPL-3.0, so this repo is as well. `chess_eval/` contains no Maia code and is MIT. I'm not affiliated with CSSLab. Games are judged by Stockfish 19, and the human positions come from the Lichess open database.
