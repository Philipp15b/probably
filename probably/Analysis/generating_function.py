import sympy

from probably.pgcl.ast import *
from probably.pgcl.syntax import check_is_linear_expr


def _is_constant_constraint(expression):  # Move this to expression checks etc.
    if isinstance(expression.lhs, VarExpr):
        if isinstance(expression.rhs, NatLitExpr):
            return True
    else:
        return False


class GeneratingFunction:
    """
    This class represents a generating function. It wraps the sympy library.
    """

    def __init__(self, function: str = ""):
        self._function = sympy.S(function, rational=True)
        self._dimension = len(self._function.free_symbols)

    def __add__(self, other):
        if isinstance(other, GeneratingFunction):
            return GeneratingFunction(self._function + other._function)
        else:
            raise SyntaxError(f"you try to add {1} with {2}", type(self), type(other))

    def __sub__(self, other):
        #raise NotImplementedError("Monus operation needs to be investigated.")
        if isinstance(other, GeneratingFunction):
            return GeneratingFunction(self._function - other._function)
        else:
            raise SyntaxError(f"you try to subtract {2} from {1}", type(self), type(other))

    def __mul__(self, other):
        if isinstance(other, GeneratingFunction):
            return GeneratingFunction(self._function * other._function)
        else:
            raise SyntaxError(f"you try to add {1} with {2}", type(self), type(other))

    def __truediv__(self, other):
        raise NotImplementedError("Division currently not supported")

    def __str__(self):
        return str(self._function)

    def __repr__(self):
        return repr(self._function)

    def __eq__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        else:
            return True if self._function-other._function == 0 else False

    def __le__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        return self == other or self < other

    def __ge__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        return self >= other

    def __lt__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        else:
            raise NotImplementedError("Not yet Implemented")

    def __gt__(self, other):
        return not self <= other

    def __ne__(self, other):
        return self != other

    def diff(self, variable, k):
        return GeneratingFunction(sympy.diff(self._function, sympy.S(variable), k))

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
            raise AssertionError("Expression has an unknown format and/or type.")

    def filter(self, expression):
        """
        Rough implementation of a filter. Can only handle distributions with finite support, or constant constraints on
        infinite support distributions.
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
                        monomial *= sympy.S(var ** state[i])
                    result += probability * monomial
            return result
        else:
            if expression.operator == Unop.NEG:
                return GeneratingFunction(self._function - self.filter(expression.expr)._function)
            if expression.operator == Binop.AND:
                result = self.filter(expression.lhs)
                return result.filter(expression.rhs)
            if expression.operator == Binop.OR:
                neg_lhs = UnopExpr(operator=Unop.NEG, expr=expression.lhs)
                neg_rhs = UnopExpr(operator=Unop.NEG, expr=expression.rhs)
                neg_conj = BinopExpr(operator=Binop.AND, lhs=neg_lhs, rhs=neg_rhs)
                neg_expr = UnopExpr(operator=Unop.NEG, expr=neg_conj)
                return self.filter(neg_expr)
            if expression.operator == Binop.LE:
                if _is_constant_constraint(expression):
                    variable = sympy.S(str(expression.lhs))
                    constant = expression.rhs.value
                    result = 0
                    for i in range(0, constant):
                        result += (sympy.diff(self._function, variable, i) / sympy.factorial(i)).subs(
                            variable, 0) * variable ** i
                    return GeneratingFunction(result)
            elif expression.operator == Binop.LEQ:
                if _is_constant_constraint(expression):
                    variable = sympy.S(str(expression.lhs))
                    constant = expression.rhs.value
                    result = 0
                    for i in range(0, constant + 1):
                        result += (sympy.diff(self._function, variable, i) / sympy.factorial(i)).subs(
                            variable, 0) * variable ** i
                    return GeneratingFunction(result)
            elif expression.operator == Binop.EQ:
                variable = sympy.S(str(expression.lhs))
                constant = expression.rhs.value
                return GeneratingFunction(
                    (sympy.diff(self._function, variable, constant) / sympy.factorial(constant))
                    .subs(variable, 0)
                    * variable ** constant
                )
            else:
                raise NotImplementedError("Infinite GFs not supported right now.")

    def linear_transformation(self, variable, expression):
        # Transform expression into sympy readable format
        rhs = sympy.S(str(expression))
        subst_var = sympy.S(str(variable))

        # Check whether the expression contains the substitution variable or not
        terms = rhs.as_coefficients_dict()
        result = self._function
        if subst_var not in terms.keys():
            result = result.subs(subst_var, 1)

        # Do the actual update stepwise
        const_correction_term = 1
        replacements = []
        for var in terms:

            # if there is a constant term, just do a multiplication
            if var == 1:
                const_correction_term = subst_var**terms[1]
            # if the variable is the substitution, a different update is necessary
            elif var == subst_var:
                replacements.append((var, subst_var**terms[var]))
            # otherwise always assume we do an addition on x
            else:
                replacements.append((var, var*subst_var**terms[var]))
        return GeneratingFunction(result.subs(replacements) * const_correction_term)
