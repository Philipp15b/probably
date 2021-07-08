import sympy

from probably.pgcl.ast import *
from probably.pgcl.syntax import check_is_linear_expr


def _is_constant_constraint(expression):  # Move this to expression checks etc.
    if isinstance(expression.lhs, VarExpr):
        if isinstance(expression.rhs, NatLitExpr):
            return True
    else:
        return False


class ComparisonException(Exception):
    pass


class GeneratingFunction:
    """
    This class represents a generating function. It wraps the sympy library.
    A GeneratingFunction object is designed to be immutable.
    This class does not ensure to be in a healthy state (i.e., every coefficient is non-negative).
    """
    def __init__(self, function: str = "", variables=set()):
        self._function = sympy.S(function, rational=True)
        self._dimension = len(self._function.free_symbols)
        self._variables = self._function.free_symbols
        for variable in variables:
            self._variables = self._variables.union({sympy.S(variable)})

    def __add__(self, other):
        if isinstance(other, GeneratingFunction):
            return GeneratingFunction(self._function + other._function, variables=self._variables)
        else:
            raise SyntaxError("you try to add" + str(type(self)) + " with " + str(type(other)))

    def __sub__(self, other):
        if isinstance(other, GeneratingFunction):
            return GeneratingFunction(self._function - other._function, variables=self._variables)
        else:
            raise SyntaxError(f"you try to subtract {2} from {1}", type(self),
                              type(other))

    def __mul__(self, other):
        if isinstance(other, GeneratingFunction):
            return GeneratingFunction(self._function * other._function, variables=self._variables)
        else:
            raise SyntaxError(f"you try to add {1} with {2}", type(self),
                              type(other))

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
            return True if self._function - other._function == 0 else False

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return not (self < other)

    def __lt__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        else:
            if self.is_finite():
                terms = self._function.as_coefficients_dic()
                for monomial in terms:
                    if terms[monomial] >= other.probability_of(monomial):
                        return False
                return True
            elif other.is_finite():
                terms = other._function.as_coefficients_dict()
                for monomial in terms:
                    if terms[monomial] >= self.probability_of(monomial):
                        return False
                return True
            else:
                difference = (self._function - other._function)
                if difference.is_polynomial():
                    return all(map(lambda x: x >= 0, difference.as_coefficients_dict().values()))
                else:
                    raise ComparisonException("Both objects have infinite support. We cannot determine the order between them.")

    def __gt__(self, other):
        return not (self <= other)

    def __ne__(self, other):
        return self != other

    def diff(self, variable, k):
        return GeneratingFunction(sympy.diff(self._function, sympy.S(variable), k), variables=self._variables)


    def as_series(self):
        if self._dimension == 1:
            return self._function.lseries()
        else:
            # TODO Tobias wants to implement this.
            # Important: make this a generator, so we can query arbitrary many terms.
            # see difference between sympy series and lseries.
            NotImplementedError("Multivariate Taylor is needed here.")

    def dim(self):
        return self._dimension

    def vars(self):
        return self._variables

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

    def evaluate(self, expression, monomial):
        operator = expression.operator

        if isinstance(expression, UnopExpr):
            if operator == Unop.NEG:
                return not self.evaluate(expression.expr, monomial)
            else:
                raise NotImplementedError(
                    "Iverson brackets are not supported.")

        elif isinstance(expression, BinopExpr):
            lhs = expression.lhs
            rhs = expression.rhs

            if operator == Binop.AND:
                return self.evaluate(lhs, monomial) and self.evaluate(rhs, monomial)
            elif operator == Binop.OR:
                return self.evaluate(lhs, monomial) or self.evaluate(rhs, monomial)
            elif operator == Binop.EQ or operator == Binop.LEQ or operator == Binop.LE:
                equation = sympy.S(str(expression))
                variable_valuations = monomial.as_powers_dict()
                for var in self._variables:
                    if var not in variable_valuations.keys():
                        equation = equation.subs(var, 0)
                    else:
                        equation = equation.subs(var, variable_valuations[var])
                return equation
            else:
                raise AssertionError("Expression must be an (in-)equation!")
        else:
            raise AssertionError(
                "Expression has an unknown format and/or type.")

    def filter(self, expression):
        """
        Rough implementation of a filter. Can only handle distributions with finite support, or constant constraints on
        infinite support distributions.
        :return:
        """
        if self.is_finite():
            result = sympy.S(0)
            addends = self._function.as_coefficients_dict()
            for monomial in addends:
                if self.evaluate(expression, monomial):
                    result += addends[monomial] * monomial
            return GeneratingFunction(result, variables=self._variables)
        else:
            if expression.operator == Unop.NEG:
                return GeneratingFunction(self._function - self.filter(expression.expr)._function, variables=self._variables)

            if expression.operator == Binop.AND:
                result = self.filter(expression.lhs)
                return result.filter(expression.rhs)
            if expression.operator == Binop.OR:
                neg_lhs = UnopExpr(operator=Unop.NEG, expr=expression.lhs)
                neg_rhs = UnopExpr(operator=Unop.NEG, expr=expression.rhs)
                neg_conj = BinopExpr(operator=Binop.AND,
                                     lhs=neg_lhs,
                                     rhs=neg_rhs)
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
                    return GeneratingFunction(result, variables=self._variables)
            elif expression.operator == Binop.LEQ:
                if _is_constant_constraint(expression):
                    variable = sympy.S(str(expression.lhs))
                    constant = expression.rhs.value
                    result = 0
                    for i in range(0, constant + 1):
                        result += (sympy.diff(self._function, variable, i) / sympy.factorial(i)).subs(
                            variable, 0) * variable ** i
                    return GeneratingFunction(result, variables=self._variables)
            elif expression.operator == Binop.EQ:
                variable = sympy.S(str(expression.lhs))
                constant = expression.rhs.value
                return GeneratingFunction(
                    (sympy.diff(self._function, variable, constant) / sympy.factorial(constant))
                    .subs(variable, 0)
                    * variable ** constant, variables=self._variables
                )
            else:
                raise NotImplementedError(
                    "Infinite GFs not supported right now.")

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
                const_correction_term = subst_var ** terms[1]
            # if the variable is the substitution, a different update is necessary
            elif var == subst_var:
                replacements.append((var, subst_var ** terms[var]))
            # otherwise always assume we do an addition on x
            else:
                replacements.append((var, var * subst_var ** terms[var]))
        return GeneratingFunction(result.subs(replacements) * const_correction_term, variables=self._variables)
