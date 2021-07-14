import functools

import sympy
from probably.pgcl.ast import *
import matplotlib.pyplot as plt


def _is_constant_constraint(expression):  # Move this to expression checks etc.
    if isinstance(expression.lhs, VarExpr):
        if isinstance(expression.rhs, NatLitExpr):
            return True
    else:
        return False


class ComparisonException(Exception):
    pass


class NotComputable(Exception):
    pass


class GeneratingFunction:
    """
    This class represents a generating function. It wraps the sympy library.
    A GeneratingFunction object is designed to be immutable.
    This class does not ensure to be in a healthy state (i.e., every coefficient is non-negative).
    """
    rational_preciseness = False

    def __init__(self, function: str = "", variables=set(), preciseness=1.0):
        self._function = sympy.S(function, rational=True)
        self._dimension = len(self._function.free_symbols)
        self._variables = self._function.free_symbols
        self._preciseness = sympy.S(str(preciseness), rational=True)
        for variable in variables:
            self._variables = self._variables.union({sympy.S(variable)})

    def __add__(self, other):
        if isinstance(other, GeneratingFunction):
            s, o = (self.coefficient_sum(), other.coefficient_sum())
            return GeneratingFunction(self._function + other._function, variables=self._variables,
                                      preciseness=(s + o)/(s/self._preciseness + o/other._preciseness))
        else:
            raise SyntaxError("you try to add" + str(type(self)) + " with " + str(type(other)))

    def __sub__(self, other):
        if isinstance(other, GeneratingFunction):
            s, o = (self.coefficient_sum(), other.coefficient_sum())
            return GeneratingFunction((self._function - other._function).simplify(), variables=self._variables,
                                      preciseness=(s + o)/(s/self._preciseness + o/other._preciseness))
        else:
            raise SyntaxError(f"you try to subtract {2} from {1}", type(self),
                              type(other))

    def __mul__(self, other):
        if isinstance(other, GeneratingFunction):
            # I am not sure whether the preciseness calculation is correct. Discussion needed
            s, o = (self.coefficient_sum(), other.coefficient_sum())
            return GeneratingFunction(self._function * other._function, variables=self._variables,
                                      preciseness=(s + o)/(s/self._preciseness + o/other._preciseness))
        else:
            raise SyntaxError("you try to multiply {} with {}".format(type(self),
                              type(other)))

    def __truediv__(self, other):
        raise NotImplementedError("Division currently not supported")

    def __str__(self):
        if GeneratingFunction.rational_preciseness:
            return str(self._function) + "\t@" + str(self._preciseness)
        else:
            return str(self._function) + "\t@" + str(self._preciseness.evalf())

    def __repr__(self):
        return repr(self._function)

    def __eq__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        else:
            # We rely on simplification of __sympy__ here. Thus, we cannot guarantee to detect equality when
            # simplification fails.
            return True if (self._function - other._function).simplify() == 0 else False

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return not (self < other)

    def __lt__(self, other):
        if not isinstance(other, GeneratingFunction):
            return False
        else:
            if self.is_finite():
                terms = self._function.as_coefficients_dict()
                for monomial in terms:
                    variables = monomial.as_powers_dict()
                    for var in variables:
                        if terms[monomial] >= other.probability_of({var: variables[var]}):
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
                    raise ComparisonException(
                        "Both objects have infinite support. We cannot determine the order between them.")

    def __gt__(self, other):
        return not (self <= other)

    def __ne__(self, other):
        return self != other

    def _monomial_to_state(self, monomial):
        variables_and_powers = monomial.as_powers_dict()
        result = dict()
        for var in self._variables:
            if var in variables_and_powers.keys():
                result[var] = variables_and_powers[var]
            else:
                result[var] = 0
        return result

    def is_precise(self):
        return self._preciseness >= 1.0

    def precision(self):
        return self._preciseness

    @classmethod
    def split_addend(cls, addend):
        prob_and_mon = addend.as_coefficients_dict()
        result = tuple()
        for monomial in prob_and_mon:
            result += (prob_and_mon[monomial],)
            result += (monomial,)
        return result

    def diff(self, variable, k):
        return GeneratingFunction(sympy.diff(self._function, sympy.S(variable), k), variables=self._variables,
                                  preciseness=self._preciseness)

    def _term_generator(self):
        assert self._function.is_polynomial(), "Terms can only be generated for finite GF"
        terms = self._function.as_coefficients_dict()
        for term in terms:
            yield sympy.S(terms[term] * term)

    def as_series(self):
        if self._function.is_polynomial():
            return self._term_generator()

        if 0 <= self._dimension <= 1:
            series = self._function.lseries()
            if str(type(series)) == "<class 'generator'>":
                return self._function.lseries()
            else:
                return {self._function}

        else:
            # TODO Tobias wants to implement this.
            # Important: make this a generator, so we can query arbitrary many terms.
            # see difference between sympy series and lseries.
            NotImplementedError("Multivariate Taylor is needed here.")

    def expand_until(self, threshold):
        if threshold > self.coefficient_sum():
            raise RuntimeError("Threshold cannot be larger than total coefficient sum! Threshold: {}, CSum {}"
                               .format(threshold, self.coefficient_sum()))
        expanded_expr = GeneratingFunction(str(0), self._variables)
        for term in self.as_series():
            if expanded_expr.coefficient_sum() >= threshold:
                break
            else:
                expanded_expr += GeneratingFunction(term, self._variables)
        expanded_expr._preciseness = self._preciseness * expanded_expr.coefficient_sum()/self.coefficient_sum()
        return expanded_expr

    def dim(self):
        return self._dimension

    def coefficient_sum(self):
        coefficient_sum = self._function
        for var in self._variables:
            coefficient_sum = coefficient_sum.limit(var, 1, "-")
        return coefficient_sum

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
            result = sympy.diff(result, sympy.S(variable), value)
            result /= sympy.factorial(value)
        for var in self.vars():
            result = result.subs(var, 0)
        return result

    def probability_mass(self):
        # use 'limit' instead of 'subs' to handle factored PGFs with denominators such as 1-x correctly
        return functools.reduce(lambda x, y: x.limit(y, 1), self.vars(), self._function)

    def normalized(self):
        mass = self.probability_mass()
        if mass == 0:
            raise ZeroDivisionError
        return GeneratingFunction(self._function / mass, preciseness=self._preciseness)

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
                if operator == Binop.EQ:
                    equation = sympy.S(f"{lhs} - {rhs}")
                else:
                    equation = sympy.S(str(expression))
                variable_valuations = monomial.as_powers_dict()
                for var in self._variables:
                    if var not in variable_valuations.keys():
                        equation = equation.subs(var, 0)
                    else:
                        equation = equation.subs(var, variable_valuations[var])
                if operator == Binop.EQ:
                    equation = equation == 0
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
        if expression.operator == Unop.NEG:
            return self - self.filter(expression.expr)
        elif expression.operator == Binop.AND:
            result = self.filter(expression.lhs)
            return result.filter(expression.rhs)
        elif expression.operator == Binop.OR:
            neg_lhs = UnopExpr(operator=Unop.NEG, expr=expression.lhs)
            neg_rhs = UnopExpr(operator=Unop.NEG, expr=expression.rhs)
            neg_conj = BinopExpr(operator=Binop.AND,
                                 lhs=neg_lhs,
                                 rhs=neg_rhs)
            neg_expr = UnopExpr(operator=Unop.NEG, expr=neg_conj)
            return self.filter(neg_expr)
        elif _is_constant_constraint(expression):
            if expression.operator == Binop.LE:
                variable = sympy.S(str(expression.lhs))
                constant = expression.rhs.value
                result = sympy.S(0)
                for i in range(0, constant):
                    result += (sympy.diff(self._function, variable, i) / sympy.factorial(i)).limit(
                        variable, 0) * variable ** i
                return GeneratingFunction(result.simplify(), variables=self._variables, preciseness=self._preciseness)
            elif expression.operator == Binop.LEQ:
                variable = sympy.S(str(expression.lhs))
                constant = expression.rhs.value
                result = sympy.S(0)
                for i in range(0, constant + 1):
                    result += (sympy.diff(self._function, variable, i) / sympy.factorial(i)).limit(
                        variable, 0) * variable ** i
                return GeneratingFunction(result.simplify(), variables=self._variables, preciseness=self._preciseness)
            elif expression.operator == Binop.EQ:
                variable = sympy.S(str(expression.lhs))
                constant = expression.rhs.value
                return GeneratingFunction(
                    ((sympy.diff(self._function, variable, constant) / sympy.factorial(constant))
                    .limit(variable, 0)
                    * variable ** constant).simplify(), variables=self._variables, preciseness=self._preciseness
                )
        elif self.is_finite():
            result = sympy.S(0)
            addends = self._function.as_coefficients_dict()
            for monomial in addends:
                if self.evaluate(expression, monomial):
                    result += addends[monomial] * monomial
            return GeneratingFunction(result.simplify(), variables=self._variables, preciseness=self._preciseness)
        else:
            raise NotComputable("Instruction {} is not computable on infinite generating function {}"
                                .format(expression, self._function))

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
        return GeneratingFunction(result.subs(replacements) * const_correction_term, variables=self._variables,
                                  preciseness=self._preciseness)

    def create_histogram(self, n=100):
        """
        Shows the encoded distribution as a histogram.
        """
        if self._dimension > 2:
            raise Exception("We can only illustrate distributions with dimension 2 or less.")

        if self._dimension > 1:
            raise NotImplementedError("Only support of dimension 1")

        if not self.is_finite():
            gf = GeneratingFunction(self._function.series(n=n).removeO(), self._variables, self.precision())
            gf.create_histogram()
        else:
            data = []
            ind = []
            for addend in self.as_series():
                (prob, mon) = self.split_addend(addend)
                state = self._monomial_to_state(mon)
                data.append(prob)
                for var in self._variables:
                    ind.append(float(state[var]))
            ax = plt.subplot()
            ax.bar(ind, data, 1, linewidth=.5, ec=(0, 0, 0))
            ax.set_xlabel('X')
            ax.set_xticks(ind)
            ax.set_ylabel('Probability p(x)')
            plt.show()
