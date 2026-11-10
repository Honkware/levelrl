# LevelRL results

Units: Lichess blitz = -803 + 1.528 x ladder (affine fit, slope 95% CI 1.369 to 1.677), snapshot data/anchors/lichess_maia3_calibration_20261110T013039Z.json. Every Elo interval below is the gauntlet's bootstrap interval carried through this fit. It leaves out the uncertainty of the fit itself, so absolute Lichess numbers are looser than the intervals read, while differences between arms at one target are not affected much.

## Anchors

| dial | ladder | Lichess | fit residual |
|---|---|---|---|
| 600 | 1089 | 890 | +29 |
| 800 | 1251 | 1111 | +3 |
| 1000 | 1371 | 1291 | -0 |
| 1200 | 1495 | 1429 | -53 |
| 1400 | 1515 | 1513 | +1 |
| 1600 | 1670 | 1695 | -53 |
| 1800 | 1725 | 1800 | -33 |
| 2000 | 1773 | 2007 | +102 |
| 2200 | 1882 | 2263 | +190 |
| 2400 | 1949 | 2159 | -16 |
| 2600 | 1979 | 2218 | -3 |

## Base conditioning curve

| dial | Lichess Elo | 95% CI | off by | games |
|---|---|---|---|---|
| 1100 | 1258 | 1213 to 1306 | +158 | 448 |
| 1300 | 1415 | 1366 to 1462 | +115 | 448 |
| 1500 | 1581 | 1521 to 1635 | +81 | 448 |
| 1700 | 1657 | 1606 to 1709 | -43 | 448 |
| 1900 | 1775 | 1724 to 1827 | -125 | 448 |

Targets: [1100, 1900].

## Maia-1 cross-check

| bot | predicted | Lichess | error |
|---|---|---|---|
| maia1 | 1534 | 1437 | +97 |
| maia5 | 1685 | 1463 | +222 |
| maia9 | 1727 | 1586 | +141 |

## T = 1100

| arm w_out/w_fid | checkpoint | Elo | 95% CI | vs T | fidelity top-1 | vs base (pp, 95% CI) | KL to base at the end |
|---|---|---|---|---|---|---|---|
| 1.0/0.0 | 50 | 1086 | 1050 to 1118 | -14 | 53.4 | -2.1 to -0.8 | 0.283 |
| 0.0/1.0 | 75 | 1266 | 1230 to 1307 | +166 | 54.9 | -0.2 to +0.3 | 0.002 |
| 0.5/0.5 | final | 1232 | 1195 to 1270 | +132 | 54.8 | -0.4 to +0.2 | 0.006 |
| 0.75/0.25 | final | 1203 | 1168 to 1240 | +103 | 55.0 | -0.2 to +0.5 | 0.013 |
| 0.25/0.75 | final | 1245 | 1209 to 1287 | +145 | 55.0 | -0.1 to +0.4 | 0.004 |

Elo during training, at each eval:

| arm | 25 | 50 | 75 | final |
|---|---|---|---|---|
| 1.0/0.0 | 1132 | 1101 | 1028 | 1175 |
| 0.0/1.0 | 1242 | 1220 | 1270 | 1302 |
| 0.5/0.5 | 1246 | 1279 | 1264 | 1189 |
| 0.75/0.25 | 1291 | 1262 | 1220 | 1257 |
| 0.25/0.75 | 1279 | 1252 | 1265 | 1274 |

Conditioning sensitivity of each final checkpoint (Elo when the dial is moved):

| arm | T-200 | T | T+200 | span |
|---|---|---|---|---|
| 1.0/0.0 | 1020 | 1099 | 1090 | +69 |
| 0.0/1.0 | 1165 | 1223 | 1347 | +182 |
| 0.5/0.5 | 1146 | 1255 | 1347 | +201 |
| 0.75/0.25 | 1143 | 1131 | 1312 | +169 |
| 0.25/0.75 | 1168 | 1242 | 1340 | +172 |

Human-likeness, outside the reward (percent of moves, Stockfish at 100k nodes):

| player | moves | blunders | of which piece drops | opening mistakes | middlegame | endgame |
|---|---|---|---|---|---|---|
| humans at 1100 | 3000 | 14.2 | 31 | 26.9 | 27.4 | 16.2 |
| base model | 8000 | 12.1 | 37 | 19.6 | 29.0 | 16.0 |
| 1.0/0.0 | 8000 | 11.9 | 37 | 20.5 | 25.8 | 13.7 |
| 0.0/1.0 | 8000 | 11.2 | 34 | 21.3 | 25.0 | 14.7 |
| 0.5/0.5 | 8000 | 12.0 | 31 | 19.1 | 26.0 | 15.8 |
| 0.75/0.25 | 8000 | 12.4 | 33 | 22.5 | 28.2 | 15.9 |
| 0.25/0.75 | 8000 | 10.8 | 34 | 21.6 | 24.9 | 13.8 |

## T = 1900

| arm w_out/w_fid | checkpoint | Elo | 95% CI | vs T | fidelity top-1 | vs base (pp, 95% CI) | KL to base at the end |
|---|---|---|---|---|---|---|---|
| 1.0/0.0 | 75 | 1899 | 1864 to 1933 | -1 | 55.2 | -5.0 to -3.5 | 0.453 |
| 0.0/1.0 | 50 | 1775 | 1739 to 1810 | -125 | 59.6 | -0.1 to +0.4 | 0.003 |
| 0.5/0.5 | 25 | 1818 | 1786 to 1856 | -82 | 59.4 | -0.4 to +0.3 | 0.005 |
| 0.75/0.25 | 75 | 1821 | 1788 to 1857 | -79 | 59.2 | -0.6 to +0.1 | 0.012 |
| 0.25/0.75 | final | 1783 | 1747 to 1818 | -117 | 59.6 | -0.1 to +0.4 | 0.004 |

Elo during training, at each eval:

| arm | 25 | 50 | 75 | final |
|---|---|---|---|---|
| 1.0/0.0 | 1952 | 1885 | 1874 | 1848 |
| 0.0/1.0 | 1780 | 1753 | 1766 | 1748 |
| 0.5/0.5 | 1763 | 1844 | 1791 | 1762 |
| 0.75/0.25 | 1841 | 1818 | 1816 | 1784 |
| 0.25/0.75 | 1778 | 1757 | 1759 | 1813 |

Conditioning sensitivity of each final checkpoint (Elo when the dial is moved):

| arm | T-200 | T | T+200 | span |
|---|---|---|---|---|
| 1.0/0.0 | 1795 | 1853 | 1927 | +132 |
| 0.0/1.0 | 1695 | 1695 | 1876 | +181 |
| 0.5/0.5 | 1735 | 1783 | 1913 | +178 |
| 0.75/0.25 | 1695 | 1813 | 1907 | +213 |
| 0.25/0.75 | 1672 | 1792 | 1902 | +230 |

Human-likeness, outside the reward (percent of moves, Stockfish at 100k nodes):

| player | moves | blunders | of which piece drops | opening mistakes | middlegame | endgame |
|---|---|---|---|---|---|---|
| humans at 1900 | 3000 | 9.1 | 20 | 14.1 | 20.8 | 10.7 |
| base model | 8000 | 7.8 | 23 | 11.3 | 21.8 | 13.6 |
| 1.0/0.0 | 8000 | 7.9 | 24 | 8.4 | 19.9 | 16.4 |
| 0.0/1.0 | 8000 | 8.4 | 25 | 10.7 | 24.3 | 13.8 |
| 0.5/0.5 | 8000 | 8.9 | 25 | 9.3 | 23.6 | 15.2 |
| 0.75/0.25 | 8000 | 8.8 | 21 | 11.7 | 23.1 | 15.3 |
| 0.25/0.75 | 8000 | 9.3 | 20 | 10.6 | 23.3 | 15.6 |

## Hack catalog

0 entries, in hack_catalog.json with example PGNs.

