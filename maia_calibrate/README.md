# maia-calibrate

Maia-3 takes a rating as input, but the strength it plays at can be 200 Elo or more away from it. This package ships the dial and sampling temperature I measured for each target rating from 600 to 2600, so you can look up the setting that actually plays at the rating you want. It is a lookup table with a small API around it. It has no dependencies and doesn't load the model.

```sh
pip install maia-calibrate
```

```python
from maia_calibrate import maia3_79m

s = maia3_79m().lookup(1500).setting
print(s.dial, s.temp)   # 1520 1.0
```

Pass `s.dial` to Maia-3 as the rating for both players and sample moves at `s.temp`.

`lookup` returns the measured setting when the target is one of the 21 that were measured (every 100 from 600 to 2600). For anything in between it returns the nearest measured setting with `validated=False`, and it doesn't interpolate.

```python
r = maia3_79m().lookup(1950)
print(r.setting.dial, r.validated, r.nearest)   # 2010 False True
```

Each entry also carries the rating it was measured at and the interval, so you can see how close it lands. The targets from 2400 up fall short, because the model tops out at about 2440.

If you measure your own settings, `CalibrationMap.from_rows`, `save_map` and `load_map` read and write the same JSON format.

The measurements and the method are in the [LevelRL](https://github.com/Honkware/levelrl) repo. Ratings are estimated Lichess blitz ratings for Maia3-79M. AGPL-3.0, and not affiliated with the CSSLab.
