"""The scaling law refuses a loss it cannot reach."""

import pytest

from mingpu.planner.define_search_space.scaling_law import BESIROGLU, UnreachableLoss


def test_unreachable_loss():
    with pytest.raises(UnreachableLoss):
        BESIROGLU.d_req(1e9, 1.9)  # a 1B model cannot reach 1.9
    with pytest.raises(UnreachableLoss):
        BESIROGLU.compute_optimal(BESIROGLU.E)  # nothing reaches the irreducible loss
