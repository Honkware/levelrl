> **Partial run:** Stopped after target 1100 by decision. Adaptive training at 1900 was not run; no result is imputed. Run 2 remains a separate historical comparison.

# LevelRL results

Code: commit d0768cb79c99cb203a1c6f3fafa25cd381f768ef.

Units: Lichess blitz = -766 + 1.496 x ladder (affine fit, slope 95% CI 1.341 to 1.643), snapshot data/anchors/lichess_maia3_calibration_20261110T071547Z.json. Elo intervals marked "in run" are the gauntlet's bootstrap interval carried through this fit, which is what compares arms with each other. Those marked "with units" add the fit's own uncertainty, which is what an absolute Lichess number carries. The fit is only tested on Maia-3 players: see the cross-check.

## Anchors

| dial | ladder | Lichess | fit residual |
|---|---|---|---|
| 600 | 1092 | 890 | +23 |
| 800 | 1241 | 1111 | +20 |
| 1000 | 1378 | 1291 | -5 |
| 1200 | 1487 | 1429 | -30 |
| 1400 | 1543 | 1513 | -30 |
| 1600 | 1682 | 1695 | -55 |
| 1800 | 1727 | 1800 | -18 |
| 2000 | 1805 | 2007 | +73 |
| 2200 | 1875 | 2263 | +223 |
| 2400 | 1937 | 2159 | +27 |
| 2600 | 1972 | 2218 | +33 |

## Base conditioning curve

| dial | ladder | Lichess Elo | 95% CI in run | with units | off by | games |
|---|---|---|---|---|---|---|
| 1100 | 1385 | 1305 | 1258 to 1353 | 1242 to 1368 | +205 | 448 |
| 1300 | 1478 | 1446 | 1400 to 1494 | 1386 to 1506 | +146 | 448 |
| 1500 | 1503 | 1483 | 1437 to 1528 | 1424 to 1542 | -17 | 448 |
| 1700 | 1595 | 1620 | 1569 to 1668 | 1556 to 1684 | -80 | 448 |
| 1900 | 1681 | 1749 | 1699 to 1797 | 1682 to 1815 | -151 | 448 |

Targets: [1100, 1900].

## Maia-1 cross-check

| bot | predicted | Lichess | error |
|---|---|---|---|
| maia1 | 1525 | 1444 | +81 |
| maia5 | 1593 | 1486 | +107 |
| maia9 | 1765 | 1592 | +173 |

## Dial 1100

The base model plays 1305 here. Each arm's goal is a share of the way to 1100, and the checkpoint reported is the last one, with no selection.

Where each arm landed:

| goal | ladder | Elo | 95% CI in run | with units | vs goal | in band, last third | rating sd, last third | outcome weight at the end | training games |
|---|---|---|---|---|---|---|---|---|---|
| 25% of the way, 1254 | 1350 | 1254 | 1221 to 1287 | 1200 to 1308 | -0 | 100% | 29 | -0.58 | 1,464 (stopped early) |
| 50% of the way, 1203 | 1328 | 1221 | 1186 to 1254 | 1165 to 1277 | +18 | 100% | 15 | -2.05 | 2,392 (stopped early) |
| 75% of the way, 1151 | 1272 | 1137 | 1099 to 1171 | 1075 to 1198 | -15 | 94% | 29 | -2.47 | 3,240 (stopped early) |
| 100% of the way, 1100 | 1269 | 1132 | 1096 to 1170 | 1070 to 1194 | +32 | 97% | 26 | -3.46 | 3,472 (stopped early) |

What the frozen model does with no training. Each is set to play this dial's Elo and judged on the same 10,000 human positions at this band. An arm has to beat these to be worth training:

| method | dial | temperature | Elo | 95% CI in run | vs dial | top-1 | vs dial as asked (pp, 95% CI) | log-loss vs it (nats, 95% CI) |
|---|---|---|---|---|---|---|---|---|
| dial as asked | 1100 | 1.0 | 1305 | 1267 to 1344 | +205 | 54.8 | +0.0 to +0.0 | +0.000 to +0.000 |
| remap | 910 | 1.0 | 1155 | 1119 to 1190 | +55 | 53.6 | -1.7 to -0.8 | +0.043 to +0.051 |
| offset | 1080 | 1.0 | 1227 | 1192 to 1266 | +127 | 54.8 | -0.2 to +0.2 | +0.001 to +0.003 |
| temperature | 1100 | 1.25 | 1078 | 1042 to 1111 | -22 | 54.8 | +0.0 to +0.0 | +0.038 to +0.046 |

What it cost in human-likeness, on 10,000 held-out human positions at this band:

| goal | Elo moved | top-1 | vs base (pp, 95% CI) | log-loss vs base (nats, 95% CI) | KL to base |
|---|---|---|---|---|---|
| 25% of the way, 1254 | -51 | 54.7 | -0.5 to +0.2 | +0.003 to +0.007 | 0.009 |
| 50% of the way, 1203 | -85 | 54.0 | -1.3 to -0.4 | +0.019 to +0.028 | 0.034 |
| 75% of the way, 1151 | -169 | 54.0 | -1.4 to -0.3 | +0.024 to +0.034 | 0.046 |
| 100% of the way, 1100 | -173 | 53.7 | -1.7 to -0.6 | +0.037 to +0.048 | 0.053 |

Band-aware pick: of the evals that measured within 50 Elo of the goal, the checkpoint with the best move-match, with its Elo from a fresh gauntlet. This one is selected, so read it beside the last checkpoint above, not in place of it.

| goal | checkpoint | Elo | 95% CI in run | vs goal | top-1 | vs base (pp, 95% CI) |
|---|---|---|---|---|---|---|
| 25% of the way, 1254 | final | 1254 | 1221 to 1287 | -0 | 54.7 | -0.5 to +0.2 |
| 50% of the way, 1203 | 50 | 1256 | 1226 to 1292 | +53 | 54.3 | -1.0 to -0.1 |
| 75% of the way, 1151 | 25 | 1152 | 1115 to 1185 | +0 | 54.4 | -0.9 to +0.1 |
| 100% of the way, 1100 | 25 | 1107 | 1071 to 1142 | +7 | 54.0 | -1.3 to -0.4 |

Elo during training, at each eval:

| goal | 25 | 50 | 75 | 100 | final |
|---|---|---|---|---|---|
| 25% of the way, 1254 | 1235 | 1298 |
| 50% of the way, 1203 | 1245 | 1198 | 1176 |
| 75% of the way, 1151 | 1196 | 1160 | 1169 | 1149 |
| 100% of the way, 1100 | 1103 | 1056 | 1055 | 1141 | 1130 |

Conditioning sensitivity of each final checkpoint (Elo when the dial is moved):

| goal | dial-200 | dial | dial+200 | span |
|---|---|---|---|---|
| 25% of the way, 1254 | 1169 | 1223 | 1324 | +156 |
| 50% of the way, 1203 | 1117 | 1180 | 1327 | +210 |
| 75% of the way, 1151 | 1057 | 1136 | 1274 | +217 |
| 100% of the way, 1100 | 1004 | 1109 | 1240 | +236 |

Human-likeness, outside the reward (percent of moves ± 95%, Stockfish at 100k nodes):

| player | moves | blunders | of which piece drops | opening mistakes | middlegame | endgame |
|---|---|---|---|---|---|---|
| humans at 1100 | 16000 | 14.9 ± 0.6 | 29 | 25.6 ± 1.2 | 28.2 ± 0.9 | 16.8 ± 1.6 |
| base model | 16000 | 11.8 ± 0.5 | 35 | 21.4 ± 1.6 | 26.9 ± 1.0 | 17.5 ± 0.9 |
| no training: remap | 16000 | 12.6 ± 0.5 | 38 | 23.6 ± 1.7 | 28.5 ± 1.0 | 16.3 ± 0.9 |
| no training: offset | 16000 | 12.0 ± 0.5 | 34 | 20.9 ± 1.6 | 25.9 ± 1.0 | 15.8 ± 0.9 |
| no training: temperature | 16000 | 13.1 ± 0.5 | 37 | 25.4 ± 1.8 | 28.6 ± 1.0 | 17.7 ± 0.9 |
| 25% of the way, 1254 | 16000 | 12.7 ± 0.5 | 33 | 20.9 ± 1.6 | 28.8 ± 1.0 | 17.9 ± 1.0 |
| 50% of the way, 1203 | 16000 | 13.0 ± 0.5 | 33 | 20.7 ± 1.6 | 29.3 ± 1.0 | 16.7 ± 1.0 |
| 75% of the way, 1151 | 16000 | 13.0 ± 0.5 | 35 | 22.4 ± 1.7 | 28.6 ± 1.0 | 17.3 ± 1.0 |
| 100% of the way, 1100 | 16000 | 12.0 ± 0.5 | 34 | 20.9 ± 1.6 | 27.3 ± 1.0 | 14.8 ± 0.9 |

## Hack catalog

2 entries, in hack_catalog.json with example PGNs.

- T1100_goal50: judge gap at eval 50
- T1100_goal75: judge gap at eval 50
