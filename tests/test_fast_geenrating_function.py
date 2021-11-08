import pytest

from probably.analysis.forward.fast_generating_function import FPS, FPSFactory


def test_geometric():
    dist = FPSFactory.geometric("x", "1/2")
    print(dist)
    assert True