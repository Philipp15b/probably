import sympy

from probably.analysis.generating_function import *
from probably.pgcl.ast import *

import pytest


def test_i_dont_know():
    gf1 = GeneratingFunction("x**2*y**3")
    gf2 = GeneratingFunction("1/2*x**2*y**3")
    assert gf1 >= gf2
