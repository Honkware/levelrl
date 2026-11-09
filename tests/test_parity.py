from __future__ import annotations

from levelrl import data
from levelrl.parity import cell


def test_every_band_sits_inside_one_figure_bin():
    for b in data.BANDS:
        assert cell(b - data.HALF) == cell(b + data.HALF), b
