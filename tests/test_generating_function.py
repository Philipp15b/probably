import sympy

from probably.analysis.generating_function import *
import probably.pgcl
import random

import pytest


def create_random_gf(vars: int = 1, terms: int = 1):
    # This does most likely does not create a PGF!
    symbols = [sympy.S("x"+str(i)) for i in range(vars)]
    values = [sympy.S(random.randint(0, 100)) for _ in range(len(symbols)*terms)]
    coeffs = [sympy.S(random.randint(0, 100)) for _ in range(terms)]

    result = sympy.S(0)
    for i in range(terms):
        monomial = sympy.S(1)
        for var in symbols:
            monomial *= var**values[i]
        result += monomial * coeffs[i]
    return GeneratingFunction(result, variables=set(symbols))


def test_arithmetic():
    #Todo implement me: addition, subtraction multiplication
    pass


def test_finite_leq():
    gf1 = GeneratingFunction("x**2*y**3")
    gf2 = GeneratingFunction("1/2*x**2*y**3")
    assert not gf1 <= gf2


def test_infinite_leq():
    gf1 = GeneratingFunction("(1-sqrt(1-x**2))/x")
    gf2 = GeneratingFunction("2/(2-x)-1")
    with pytest.raises(ComparisonException):
        assert gf1 <= gf2


def test_finite_le():
    gf1 = GeneratingFunction("x**2*y**3")
    gf2 = GeneratingFunction("1/2*x**2*y**3")
    assert not gf1 < gf2


def test_infinite_le():
    gf1 = GeneratingFunction("(1-sqrt(1-x**2))/x")
    gf2 = GeneratingFunction("2/(2-x)-1")
    with pytest.raises(ComparisonException):
        assert gf1 < gf2


def test_split_addend():
    probability = sympy.S(random.random())
    number_of_vars = random.randint(1, 10)
    values = [random.randint(1, 5000) for _ in range(number_of_vars)]
    monomial = sympy.S(1)
    for i in range(number_of_vars):
        monomial *= sympy.S("x"+str(i))**values[i]
    addend = probability*monomial
    assert GeneratingFunction.split_addend(addend) == (probability, monomial)


def test_linear_transformation():
    gf = GeneratingFunction("1/2*x*c + 1/4 * x**2 + 1/4")
    gf = gf.linear_transformation("x", "4 * x + 7*c + 2")
    assert gf == GeneratingFunction("1/2*x**13*c + 1/4 * x**10 + 1/4*x**2")


def test_filter():
    # check filter on finite GF
    gf = GeneratingFunction("1/2*x*c + 1/4 * x**2 + 1/4")
    assert gf.filter(probably.pgcl.parse_expr("x*c < 150")) == gf

    # check filter on infinite GF
    gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
    assert gf.filter(probably.pgcl.parse_expr("c <= 5")) == GeneratingFunction("c/2 + c**3/8 + c**5/16")

    # check non-const filter on infinite GF:
    with pytest.raises(NotComputableException):
        gf.filter(probably.pgcl.parse_expr("x*z <= 10"))


def test_expand_until():
    gf = GeneratingFunction("2/(2-x) - 1")
    assert list(gf.expand_until(0.99))[-1] == GeneratingFunction("1/2*x + 1/4*x**2 + 1/8 * x**3 + 1/16 * x**4 + 1/32 * x**5 + 1/64 * x**6 + 1/128 * x**7")

    with pytest.raises(RuntimeError):
        gf.expand_until(random.randint(2, 10))