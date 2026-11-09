from __future__ import annotations

import torch

from levelrl.anchors import nucleus_need


def kept(p: torch.Tensor, top_p: float) -> set[int]:
    # The keep rule from maia3.uci.sample_from_logits, as written there.
    sp, si = torch.sort(p, descending=True)
    keep = torch.cumsum(sp, -1) <= top_p
    keep[0] = True
    return set(si[keep].tolist())


def test_nucleus_need_matches_maia3s_keep_rule():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        p = torch.softmax(torch.randn(40, generator=g) * 2, -1)
        need = nucleus_need(p[None].expand(40, -1), torch.arange(40))
        for q in (0.5, 0.8, 0.9, 0.95):
            assert {i for i in range(40) if need[i] <= q} == kept(p, q)
