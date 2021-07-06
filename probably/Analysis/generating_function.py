import sympy

from probably.pgcl.ast import *

class GeneratingFunction:

    """
    This class represents a generating function. It wraps the sympy library.
    """

    def __init__(self, function: str = ""):
        self._function = sympy.S(function)
        self._dimension = len(self._function.free_symbols)

    def __add__(self, other):
        result = self._function + other
        return GeneratingFunction(result)

    def __sub__(self, other):
        result = self._function - other
        return GeneratingFunction(result)

    def __mul__(self, other):
        result = self._function * other
        return GeneratingFunction(result)

    def __truediv__(self, other):
        result = self._function / other
        return GeneratingFunction(result)

    def __str__(self):
        return str(self._function)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        return self == other

    def __le__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        return self <= other

    def __ge__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        return self >= other

    def __lt__(self, other):
        return not self >= other

    def __gt__(self, other):
        return not self <= other

    def __ne__(self, other):
        return self != other

    def as_series(self):
        return self._function.lseries()

    def dim(self):
        return self._dimension

    def vars(self):
        return self._function.free_symbols

    def expected_value_of(self, variable: str):
        result = sympy.diff(self._function, sympy.S(variable))
        return sympy.limit(result, sympy.S(variable), 1, '-')

    def probability_of(self, state: dict):
        """
        Determines the probability of a single program state encoded by a monomial (discrete only).
        :param state: The queried program state.
        :return: The probability for that state.
        """
        result = self._function
        for variable, value in state.items():
            result = sympy.diff(result, sympy.symbols(variable), value)
            result /= sympy.factorial(value)
        for var in self.vars():
            result = result.subs(var, 0)
        return result

    def is_finite(self):
        """
        Checks whether the generating function is finite.
        :return: True if the GF is a polynomial, False otherwise.
        """
        return self._function.is_polynomial()

    def evaluate(self, expression, term):
        operator = expression.operator

        if isinstance(expression, UnopExpr):
            if operator == Unop.NEG:
                return not self.evaluate(expression.expr, term)
            else:
                raise NotImplementedError("Iverson brackets are not supported.")

        elif isinstance(expression, BinopExpr):
            lhs = expression.lhs
            rhs = expression.rhs

            if operator == Binop.AND:
                return self.evaluate(lhs, term) and self.evaluate(rhs, term)
            elif operator == Binop.OR:
                return self.evaluate(lhs, term) or self.evaluate(rhs, term)
            elif operator == Binop.EQ or operator == Binop.LEQ or operator == Binop.LE:
                equation = sympy.S(str(expression))
                state, probability = term
                i = 0
                for var in self._function.free_symbols:
                    equation = equation.subs(var, state[i])
                    i += 1
                return equation
            else:
                raise AssertionError("Expression must be an (in-)equation!")
        else:
            raise AssertionError("Expression has an unkown format and/or type.")

    def filter(self, expression):
        """
        TODO. This function filters a GF for a given guard.
        :return:
        """
        if self.is_finite():
            result = GeneratingFunction(0)
            for term in self._function.as_poly().terms():
                if self.evaluate(expression, term):
                    state, probability = term
                    monomial = 1
                    i = 0
                    for var in self._function.free_symbols:
                        monomial *= sympy.S(var**state[i])
                    result += probability * monomial
            return result
        else:
            raise NotImplementedError("Infinite GFs not supported right now.")



