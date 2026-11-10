# LevelRL results

Units: Lichess blitz = -759 + 1.491 x ladder (affine fit, slope 95% CI 1.319 to 1.648), snapshot data/anchors/lichess_maia3_calibration_20261109T184408Z.json. Every Elo interval below is the gauntlet's bootstrap interval carried through this fit. It leaves out the uncertainty of the fit itself, so absolute Lichess numbers are looser than the intervals read, while differences between arms at one target are not affected much.

## Anchors

| dial | ladder | Lichess | fit residual |
|---|---|---|---|
| 600 | 1050 | 890 | +84 |
| 800 | 1318 | 1111 | -96 |
| 1000 | 1369 | 1291 | +9 |
| 1200 | 1482 | 1436 | -16 |
| 1400 | 1552 | 1513 | -43 |
| 1600 | 1667 | 1695 | -32 |
| 1800 | 1697 | 1800 | +28 |
| 2000 | 1825 | 2007 | +45 |
| 2200 | 1886 | 2263 | +210 |
| 2400 | 1953 | 2159 | +5 |
| 2600 | 1990 | 2218 | +10 |

## Base conditioning curve

| dial | Lichess Elo | 95% CI | off by | games |
|---|---|---|---|---|
| 1100 | 1257 | 1188 to 1323 | +157 | 248 |
| 1300 | 1415 | 1353 to 1482 | +115 | 248 |
| 1500 | 1494 | 1429 to 1561 | -6 | 248 |
| 1700 | 1589 | 1519 to 1653 | -111 | 248 |
| 1900 | 1793 | 1728 to 1857 | -107 | 248 |

Targets: [1500, 1100].

## Maia-1 cross-check

| bot | predicted | Lichess | error |
|---|---|---|---|
| maia1 | 1441 | 1404 | +37 |
| maia5 | 1621 | 1434 | +187 |
| maia9 | 1665 | 1544 | +121 |

## T = 1500

| arm w_out/w_fid | checkpoint | Elo | 95% CI | vs T | fidelity top-1 | vs base (pp, 95% CI) |
|---|---|---|---|---|---|---|
| 1.0/0.0 | final | 1507 | 1461 to 1553 | +7 | 55.1 | -2.4 to -1.1 |
| 0.0/1.0 | 75 | 1565 | 1512 to 1618 | +65 | 54.9 | -2.4 to -1.3 |
| 0.5/0.5 | 75 | 1510 | 1460 to 1562 | +10 | 55.0 | -2.3 to -1.2 |
| 0.75/0.25 | final | 1535 | 1489 to 1590 | +35 | 55.5 | -1.8 to -0.7 |
| 0.25/0.75 | 25 | 1572 | 1525 to 1619 | +72 | 56.0 | -1.2 to -0.4 |

Elo during training, at each eval:

| arm | 25 | 50 | 75 | final |
|---|---|---|---|---|
| 1.0/0.0 | 1720 | 1470 | 1602 | 1480 |
| 0.0/1.0 | 1640 | 1667 | 1516 | 1670 |
| 0.5/0.5 | 1616 | 1580 | 1507 | 1590 |
| 0.75/0.25 | 1650 | 1463 | 1668 | 1495 |
| 0.25/0.75 | 1590 | 1646 | 1616 | 1602 |

Conditioning sensitivity of each final checkpoint (Elo when the dial is moved):

| arm | T-200 | T | T+200 | span |
|---|---|---|---|---|
| 1.0/0.0 | 1413 | 1532 | 1600 | +187 |
| 0.0/1.0 | 1433 | 1653 | 1665 | +233 |
| 0.5/0.5 | 1410 | 1560 | 1589 | +178 |
| 0.75/0.25 | 1452 | 1502 | 1647 | +195 |
| 0.25/0.75 | 1466 | 1566 | 1699 | +234 |

Human-likeness, outside the reward (percent of moves, Stockfish at 100k nodes):

| player | moves | blunders | of which piece drops | opening mistakes | middlegame | endgame |
|---|---|---|---|---|---|---|
| humans at 1500 | 3000 | 11.4 | 28 | 16.5 | 25.8 | 15.4 |
| base model | 8000 | 10.8 | 27 | 16.9 | 26.8 | 16.1 |
| 1.0/0.0 | 8000 | 9.4 | 30 | 14.1 | 24.3 | 13.9 |
| 0.0/1.0 | 8000 | 9.5 | 40 | 11.9 | 24.0 | 12.0 |
| 0.5/0.5 | 8000 | 10.5 | 34 | 12.2 | 24.1 | 14.8 |
| 0.75/0.25 | 8000 | 9.3 | 32 | 14.3 | 22.5 | 13.7 |
| 0.25/0.75 | 8000 | 10.2 | 31 | 13.6 | 24.5 | 15.6 |

## T = 1100

| arm w_out/w_fid | checkpoint | Elo | 95% CI | vs T | fidelity top-1 | vs base (pp, 95% CI) |
|---|---|---|---|---|---|---|
| 1.0/0.0 | 25 | 1107 | 1060 to 1154 | +7 | 54.4 | -0.9 to +0.1 |
| 0.0/1.0 | 25 | 1358 | 1303 to 1416 | +258 | 54.3 | -0.9 to -0.1 |
| 0.5/0.5 | 25 | 1146 | 1098 to 1194 | +46 | 54.2 | -1.1 to -0.2 |
| 0.75/0.25 | final | 1079 | 1030 to 1128 | -21 | 53.9 | -1.5 to -0.4 |
| 0.25/0.75 | final | 1273 | 1221 to 1334 | +173 | 53.2 | -2.1 to -1.0 |

Elo during training, at each eval:

| arm | 25 | 50 | 75 | final |
|---|---|---|---|---|
| 1.0/0.0 | 1096 | 1235 | 991 | 1197 |
| 0.0/1.0 | 1394 | 1358 | 1394 | 1455 |
| 0.5/0.5 | 1129 | 1248 | 1167 | 1176 |
| 0.75/0.25 | 1159 | 1046 | 1253 | 1087 |
| 0.25/0.75 | 1299 | 1282 | 1287 | 1243 |

Conditioning sensitivity of each final checkpoint (Elo when the dial is moved):

| arm | T-200 | T | T+200 | span |
|---|---|---|---|---|
| 1.0/0.0 | 948 | 1054 | 1167 | +218 |
| 0.0/1.0 | 1182 | 1393 | 1471 | +288 |
| 0.5/0.5 | 1104 | 1192 | 1305 | +201 |
| 0.75/0.25 | 981 | 1110 | 1224 | +243 |
| 0.25/0.75 | 1201 | 1379 | 1423 | +222 |

Human-likeness, outside the reward (percent of moves, Stockfish at 100k nodes):

| player | moves | blunders | of which piece drops | opening mistakes | middlegame | endgame |
|---|---|---|---|---|---|---|
| humans at 1100 | 3000 | 14.5 | 31 | 26.4 | 26.7 | 18.6 |
| base model | 8000 | 12.0 | 33 | 18.9 | 25.5 | 16.6 |
| 1.0/0.0 | 8000 | 11.1 | 40 | 23.3 | 23.1 | 14.1 |
| 0.0/1.0 | 8000 | 11.0 | 33 | 18.7 | 24.4 | 14.6 |
| 0.5/0.5 | 8000 | 10.3 | 38 | 19.2 | 24.1 | 13.2 |
| 0.75/0.25 | 8000 | 10.9 | 38 | 20.2 | 23.5 | 14.7 |
| 0.25/0.75 | 8000 | 9.8 | 37 | 16.5 | 21.4 | 13.6 |

## Hack catalog

8 entries, in hack_catalog.json with example PGNs.

- T1500_out0.0_fid1.0: judge gap at eval 75
- T1500_out0.5_fid0.5: judge gap at eval 50
- T1500_out0.25_fid0.75: judge gap at eval 50
- T1500_out0.25_fid0.75: judge gap at eval 75
- T1100_out0.0_fid1.0: judge gap at eval 75
- T1100_out0.5_fid0.5: judge gap at eval 75
- T1100_out0.75_fid0.25: threefold drift, 0.080 to 0.150
- T1100_out0.25_fid0.75: judge gap at eval 75
