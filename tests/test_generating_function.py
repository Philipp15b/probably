import sympy

from probably.analysis.generating_function import *
import probably.pgcl
from probably.pgcl.parser import _parse_instr
import random

import pytest

from probably.analysis.pgfs import PGFS


def create_random_gf(vars: int = 1, terms: int = 1):
    # This does most likely does not create a PGF!
    symbols = [sympy.S("x" + str(i)) for i in range(vars)]
    values = [sympy.S(random.randint(0, 100)) for _ in range(len(symbols) * terms)]
    coeffs = [sympy.S(str(random.uniform(0, 1)), rational=True) for _ in range(terms)]

    result = sympy.S(0)
    for i in range(terms):
        monomial = sympy.S(1)
        for var in symbols:
            monomial *= var ** values[i]
        result += monomial * coeffs[i]
    return GeneratingFunction(result, *symbols)


class TestDistributionInterface:

    def test_arithmetic(self):
        # Todo implement me: addition, subtraction multiplication
        pass

    def test_finite_leq(self):
        gf1 = GeneratingFunction("x**2*y**3")
        gf2 = GeneratingFunction("1/2*x**2*y**3")
        assert not gf1 <= gf2

    def test_infinite_leq(self):
        gf1 = GeneratingFunction("(1-sqrt(1-x**2))/x")
        gf2 = GeneratingFunction("2/(2-x)-1")
        with pytest.raises(ComparisonException):
            assert gf1 <= gf2

    def test_finite_le(self):
        gf1 = GeneratingFunction("x**2*y**3")
        gf2 = GeneratingFunction("1/2*x**2*y**3")
        assert not gf1 < gf2

    def test_infinite_le(self):
        gf1 = GeneratingFunction("(1-sqrt(1-x**2))/x")
        gf2 = GeneratingFunction("2/(2-x)-1")
        with pytest.raises(ComparisonException):
            assert gf1 < gf2

    def test_iteration(self):
        gf = GeneratingFunction("(1-sqrt(1-x**2))/x")
        expected_terms = [("1/2", {"x": 1}), ("1/8", {"x": 3}), ("1/16", {"x": 5}), ("5/128", {"x": 7}),
                          ("7/256", {"x": 9}), ("21/1024", {"x": 11}), ("33/2048", {"x": 13}), ("429/32768", {"x": 15}),
                          ("715/65536", {"x": 17}), ("2431/262144", {"x": 19})]
        i = 0
        for prob, state in gf:
            if i >= 4:
                break
            if prob == "0":
                continue
            else:
                assert (prob, state) == expected_terms[i]
                i += 1

    def test_copy(self):
        gf = create_random_gf(3, 5)
        assert gf.copy() == gf

    def test_get_probability_of(self):
        gf = GeneratingFunction("(1-sqrt(1-x**2))/x")
        assert gf.get_probability_of(probably.pgcl.parse_expr("x <= 3")) == probably.pgcl.parse_expr("5/8")

        gf = PGFS.zero("x")
        assert gf.get_probability_of(probably.pgcl.parse_expr("not (z*y < 12)")) == probably.pgcl.parse_expr("0")

    def test_get_probability_mass(self):
        gf = GeneratingFunction("(1-sqrt(1-x**2))/x")
        assert gf.get_probability_mass() == probably.pgcl.parse_expr("1")

        gf = PGFS.zero("x")
        assert gf.get_probability_mass() == probably.pgcl.parse_expr("0")

        gf = PGFS.uniform("x", "3", "10")
        assert gf.get_probability_mass() == probably.pgcl.parse_expr("1")

    def test_expected_value_of(self):
        gf = GeneratingFunction("(1-sqrt(1-x**2))/x")
        assert gf.get_expected_value_of("x") == probably.pgcl.parse_expr("\\infty")

        gf = PGFS.zero("x")
        assert gf.get_expected_value_of("x") == probably.pgcl.parse_expr("0")

        gf = PGFS.uniform("x", "3", "10")
        assert gf.get_expected_value_of("x**2+y") == probably.pgcl.parse_expr("13/2")

    def test_normalize(self):
        assert create_random_gf().normalize().coefficient_sum() == 1

    def test_get_variables(self):
        """Returns the free variables of the distribution. """
        pass

    def test_get_parameters(self):
        """ Returns the parameters of the distribution. """
        pass

    def test_filter(self):
        gf = PGFS.zero("x")
        assert gf.filter(probably.pgcl.parse_expr("x*3 < 25*y")) == gf

        # check filter on infinite GF
        gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
        assert gf.filter(probably.pgcl.parse_expr("c <= 5")) == GeneratingFunction("c/2 + c**3/8 + c**5/16")

        # check filter on finite GF
        gf = GeneratingFunction("1/2*x*c + 1/4 * x**2 + 1/4")
        assert gf.filter(probably.pgcl.parse_expr("x*c < 123")) == gf

        gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
        assert gf.filter(probably.pgcl.parse_expr("x*z <= 10")) == PGFS.zero()

    def test_is_zero_dist(self):
        gf = create_random_gf(4, 10)
        assert (gf == PGFS.zero(*gf.get_variables())) == gf.is_zero_dist()

        gf = PGFS.zero("x")
        assert (gf == PGFS.zero(*gf.get_variables())) == gf.is_zero_dist()

    def test_is_finite(self):
        gf = create_random_gf(10, 10)
        assert gf.is_finite()

        gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
        assert not gf.is_finite()

        gf = GeneratingFunction("1", "x")
        assert gf.is_finite()

    def test_update(self):
        gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
        assert gf.update(BinopExpr(Binop.EQ, VarExpr('c'), BinopExpr(Binop.PLUS, VarExpr('c'), NatLitExpr(1)))) == \
               GeneratingFunction("c*(1-sqrt(1-c**2))/c")

        gf = PGFS.zero("x")
        expr = BinopExpr(Binop.EQ,
                         VarExpr('x'),
                         BinopExpr(Binop.PLUS,
                                   BinopExpr(Binop.TIMES,
                                             VarExpr('x'),
                                             NatLitExpr(5)
                                             ),
                                   NatLitExpr(1)
                                   )
                         )
        assert gf.update(expr) == PGFS.zero("x")

        gf = PGFS.uniform("x", "0", "5")
        expr = BinopExpr(Binop.EQ, VarExpr('x'), BinopExpr(Binop.TIMES, VarExpr('x'), VarExpr('x')))
        assert gf.update(expr) == GeneratingFunction("1/6 * (1 + x + x**4 + x**9 + x**16 + x**25)")

    def test_marginal(self):
        gf = GeneratingFunction("(1-sqrt(1-c**2))/c")
        assert gf.marginal("x") == GeneratingFunction('1', 'x')

        gf = PGFS.uniform("x", '0', '10') * PGFS.binomial('y', n='10', p='1/2')
        assert gf.marginal('x') == PGFS.uniform("x", '0', '10')
        assert gf.marginal('x', method=MarginalType.Exclude) == PGFS.binomial('y', n='10', p='1/2')
        assert gf.marginal('x','y') == gf

    def test_set_variables(self):
        gf = create_random_gf(3, 5)
        gf = gf.set_variables("a", "b", "c")
        assert all([x in gf.get_variables() for x in {'a', 'b', 'c'}])

    def test_approximate(self):
        gf = GeneratingFunction("2/(2-x) - 1")
        assert list(gf.approximate("0.99"))[-1] == GeneratingFunction("1/2*x + 1/4*x**2 + 1/8 * x**3 + 1/16 * x**4"
                                                                      "+ 1/32 * x**5 + 1/64 * x**6 + 1/128 * x**7")

        gf = PGFS.zero("x", 'y')
        for prob, state in gf.approximate(10):
            assert prob == "0" and state == dict()


def test_split_addend():
    probability = sympy.S(random.random())
    number_of_vars = random.randint(1, 10)
    values = [random.randint(1, 5000) for _ in range(number_of_vars)]
    monomial = sympy.S(1)
    for i in range(number_of_vars):
        monomial *= sympy.S("x" + str(i)) ** values[i]
    addend = probability * monomial
    assert GeneratingFunction.split_addend(addend) == (probability, monomial)


def test_linear_transformation():
    gf = GeneratingFunction("1/2*x*c + 1/4 * x**2 + 1/4")
    gf = gf.linear_transformation("x", "4 * x + 7*c + 2")
    assert gf == GeneratingFunction("1/2*x**13*c + 1/4 * x**10 + 1/4*x**2")
