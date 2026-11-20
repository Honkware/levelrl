# Maia-3 full-dial calibration

Settings are selected on validation games and validation human positions. Final games use fresh openings; the 10,000-position fidelity set is only read after selection. Ratings are estimated Lichess blitz strength, not official Lichess ratings.

| target | selected method | dial | temperature | measured strength | 95% CI in run | with units | error | top-1 | log-loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 600 | temperature | 600 | 1.232 | 596 | 557 to 636 | 496 to 697 | -4 | nan | nan |
| 700 | repair | 610 | 1.087 | 748 | 710 to 788 | 659 to 836 | +48 | 49.0 | 1.683 |
| 800 | repair | 650 | 1.04 | 852 | 814 to 887 | 772 to 931 | +52 | 49.9 | 1.608 |
| 900 | repair | 840 | 1.071 | 991 | 952 to 1034 | 919 to 1063 | +91 | 51.6 | 1.529 |
| 1000 | remap | 740 | 1 | 987 | 948 to 1024 | 916 to 1058 | -13 | nan | nan |
| 1100 | temperature | 1100 | 1.297 | 1061 | 1023 to 1098 | 994 to 1127 | -39 | 54.8 | 1.460 |
| 1200 | remap | 990 | 1 | 1209 | 1174 to 1245 | 1152 to 1267 | +9 | nan | nan |
| 1300 | temperature | 1300 | 1.177 | 1282 | 1247 to 1319 | 1227 to 1337 | -18 | 55.8 | 1.385 |
| 1400 | remap | 1250 | 1 | 1382 | 1346 to 1416 | 1330 to 1434 | -18 | nan | nan |
| 1500 | remap | 1520 | 1 | 1532 | 1497 to 1567 | 1481 to 1584 | +32 | 57.1 | 1.323 |
| 1600 | remap | 1600 | 1 | 1557 | 1523 to 1593 | 1505 to 1609 | -43 | nan | nan |
| 1700 | temperature | 1700 | 0.996 | 1666 | 1628 to 1704 | 1609 to 1722 | -34 | 56.1 | 1.304 |
| 1800 | remap | 1910 | 1 | 1767 | 1734 to 1803 | 1709 to 1825 | -33 | nan | nan |
| 1900 | combined | 2010 | 0.895 | 1914 | 1879 to 1947 | 1848 to 1980 | +14 | 59.5 | 1.231 |
| 2000 | combined | 2130 | 0.876 | 1973 | 1939 to 2007 | 1903 to 2042 | -27 | nan | nan |
| 2100 | remap | 2600 | 1 | 2058 | 2025 to 2092 | 1983 to 2133 | -42 | nan | nan |
| 2200 | combined | 2400 | 0.763 | 2157 | 2124 to 2190 | 2075 to 2239 | -43 | nan | nan |
| 2300 | temperature | 2300 | 0.344 | 2347 | 2319 to 2377 | 2252 to 2443 | +47 | nan | nan |
| 2400 | repair | 2460 | 0.51 | 2322 | 2290 to 2353 | 2228 to 2417 | -78 | 62.0 | 1.369 |
| 2500 | temperature | 2500 | 0.1 | 2415 | 2381 to 2449 | 2312 to 2518 | -85 | nan | nan |
| 2600 | temperature | 2600 | 0.1 | 2443 | 2409 to 2476 | 2338 to 2548 | -157 | nan | nan |

Human fidelity is available at 700, 800, 900, 1100, 1300, 1500, 1700, 1900, 2400; other rows report strength only.

## Sampled-game human-likeness

Percent of moves, with normal-approximation 95% intervals. These intervals do not include game-to-game correlation.

| target | player | blunders | opening mistakes | middlegame | endgame |
|---|---|---|---|---|---|
| 700 | repair | 15.1 ± 0.6 | 34.9 ± 2.0 | 30.6 ± 1.1 | 17.0 ± 0.9 |
| 700 | base | 13.8 ± 0.5 | 29.2 ± 1.8 | 27.6 ± 1.0 | 14.8 ± 0.9 |
| 700 | human | 19.1 ± 1.2 | 35.4 ± 2.3 | 29.7 ± 1.9 | 17.1 ± 3.4 |
| 800 | repair | 14.0 ± 0.5 | 32.2 ± 1.8 | 28.5 ± 1.0 | 16.5 ± 0.9 |
| 800 | base | 12.7 ± 0.5 | 26.2 ± 1.7 | 25.1 ± 0.9 | 13.0 ± 0.9 |
| 800 | human | 18.8 ± 0.8 | 35.1 ± 1.6 | 31.2 ± 1.3 | 15.1 ± 2.0 |
| 900 | repair | 13.5 ± 0.5 | 27.8 ± 1.7 | 27.1 ± 1.0 | 15.0 ± 0.9 |
| 900 | base | 12.0 ± 0.5 | 23.3 ± 1.5 | 24.5 ± 0.9 | 13.2 ± 0.9 |
| 900 | human | 16.9 ± 0.6 | 30.0 ± 1.2 | 30.1 ± 1.0 | 16.2 ± 1.6 |
| 1100 | temperature | 13.4 ± 0.5 | 26.7 ± 1.8 | 29.8 ± 1.0 | 16.3 ± 0.9 |
| 1100 | base | 12.0 ± 0.5 | 21.4 ± 1.6 | 27.1 ± 1.0 | 18.2 ± 0.9 |
| 1100 | human | 14.5 ± 1.3 | 26.2 ± 2.8 | 26.8 ± 2.2 | 19.1 ± 3.7 |
| 1300 | temperature | 12.6 ± 0.5 | 20.4 ± 1.6 | 29.8 ± 1.0 | 15.9 ± 0.9 |
| 1300 | base | 11.3 ± 0.5 | 18.9 ± 1.6 | 27.3 ± 1.0 | 16.7 ± 0.9 |
| 1300 | human | 13.9 ± 1.2 | 20.5 ± 2.7 | 27.3 ± 2.1 | 17.4 ± 3.6 |
| 1500 | remap | 9.9 ± 0.5 | 15.1 ± 1.5 | 25.6 ± 1.0 | 14.0 ± 0.9 |
| 1500 | base | 10.6 ± 0.5 | 14.8 ± 1.4 | 25.6 ± 1.0 | 15.3 ± 0.9 |
| 1500 | human | 11.6 ± 1.1 | 16.5 ± 2.6 | 26.6 ± 2.1 | 15.1 ± 3.4 |
| 1700 | temperature | 9.1 ± 0.4 | 13.7 ± 1.4 | 23.9 ± 1.0 | 14.7 ± 0.9 |
| 1700 | base | 10.0 ± 0.5 | 13.7 ± 1.4 | 24.9 ± 1.0 | 14.9 ± 0.9 |
| 1700 | human | 11.5 ± 1.1 | 16.8 ± 2.6 | 24.4 ± 2.0 | 15.4 ± 3.4 |
| 1900 | combined | 8.0 ± 0.4 | 8.8 ± 1.2 | 21.2 ± 1.0 | 13.8 ± 0.8 |
| 1900 | base | 8.8 ± 0.4 | 9.5 ± 1.3 | 23.9 ± 1.0 | 15.1 ± 0.9 |
| 1900 | human | 9.0 ± 1.0 | 13.0 ± 2.4 | 21.3 ± 1.9 | 11.7 ± 3.1 |
| 2400 | repair | 4.7 ± 0.3 | 3.4 ± 0.8 | 14.3 ± 0.9 | 10.3 ± 0.7 |
| 2400 | base | 6.3 ± 0.4 | 6.3 ± 1.1 | 19.4 ± 1.0 | 12.4 ± 0.7 |
| 2400 | human | 5.6 ± 0.5 | 4.7 ± 0.9 | 15.7 ± 0.9 | 11.7 ± 1.8 |

## Validation Pareto frontiers

### 600

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| temperature | 600 | 1.232 | 627 | nan | yes |

### 700

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| combined | 650 | 1.127 | 758 | nan |  |

### 800

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| remap | 650 | 1 | 784 | nan |  |

### 900

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| combined | 800 | 1.111 | 934 | nan |  |

### 1000

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| remap | 740 | 1 | 985 | nan | yes |

### 1100

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| dial as asked | 1100 | 1 | 1319 | 1.418 |  |
| temperature | 1100 | 1.297 | 1109 | 1.462 | yes |

### 1200

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| remap | 990 | 1 | 1218 | nan | yes |

### 1300

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| dial as asked | 1300 | 1 | 1383 | 1.389 |  |
| temperature | 1300 | 1.177 | 1294 | 1.406 | yes |
| combined | 1190 | 1.089 | 1305 | 1.408 |  |

### 1400

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| remap | 1250 | 1 | 1402 | nan | yes |

### 1500

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| remap | 1520 | 1 | 1519 | 1.356 | yes |

### 1600

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| remap | 1600 | 1 | 1599 | nan | yes |

### 1700

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| temperature | 1700 | 0.996 | 1694 | 1.291 | yes |

### 1800

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| remap | 1910 | 1 | 1781 | nan | yes |

### 1900

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| dial as asked | 1900 | 1 | 1740 | 1.199 |  |
| temperature | 1900 | 0.791 | 1885 | 1.224 |  |
| combined | 2010 | 0.895 | 1922 | 1.207 | yes |

### 2000

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| combined | 2130 | 0.876 | 1979 | nan | yes |

### 2100

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| remap | 2600 | 1 | 2097 | nan | yes |

### 2200

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| combined | 2400 | 0.763 | 2176 | nan | yes |

### 2300

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| temperature | 2300 | 0.344 | 2297 | nan | yes |

### 2400

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| combined | 2500 | 0.55 | 2323 | nan |  |

### 2500

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| temperature | 2500 | 0.1 | 2429 | nan | yes |

### 2600

| method | dial | temperature | validation strength | validation log-loss | selected |
|---|---|---:|---:|---:|---|
| temperature | 2600 | 0.1 | 2479 | nan | yes |

