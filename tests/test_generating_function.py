import sympy

from probably.analysis.generating_function import *
from probably.pgcl.ast import *
import random

import pytest


def create_random_gf(vars:int = 1, terms:int = 1):
    symbols = [sympy.S("x"+str(i)) for i in range(vars)]
    values = [sympy.S(random.randint(0,5000)) for _ in range(len(symbols)*terms)]
    coeffs = [sympy.S(random.randint(0,5000)) for _ in range(terms)]

    result = sympy.S(0)
    for i in range(terms):
        monomial = sympy.S(1)
        for var in symbols:
            monomial *= symbols[i]**values[i]
        result += monomial * coeffs[i]
    return GeneratingFunction(result)


def test_finite_geq():
    gf1 = GeneratingFunction("x**2*y**3")
    gf2 = GeneratingFunction("1/2*x**2*y**3")
    assert gf1 >= gf2


@pytest.mark.xfail(raises=ComparisonException)
def test_infinite_geq():
    gf1 = GeneratingFunction("(1-sqrt(1-x))/x")
    gf2 = GeneratingFunction("2/(2-x)-1")
    assert gf1 >= gf2


def test_split_addend():
    probability = sympy.S(random.random())
    number_of_vars = random.randint(1, 10)
    values = [random.randint(1, 5000) for _ in range(number_of_vars)]
    monomial = sympy.S(1)
    for i in range(number_of_vars):
        monomial *= sympy.S("x"+str(i))**values[i]
    addend = probability*monomial
    assert GeneratingFunction.split_addend(addend) == (probability, monomial)
