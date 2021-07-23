import functools
import logging

import sympy
import matplotlib.pyplot as plt
import operator
from probably.pgcl import Unop, VarExpr, NatLitExpr, UnopExpr, BinopExpr, Binop, Expr
from .exceptions import ComparisonException, NotComputable, ParameterError

logger = logging.getLogger("probably.analysis.generating_function")
logger.setLevel(logging.INFO)
fhandler = logging.FileHandler(filename='test.log', mode='a')
fhandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(fhandler)


def _is_constant_constraint(expression):  # Move this to expression checks etc.
    if isinstance(expression.lhs, VarExpr):
        if isinstance(expression.rhs, NatLitExpr):
            return True
    else:
        return False


@functools.cache
def _enumerate_monomials(variables, max_degree):
    if variables == 0:
        return [[]]
    t = []
    for i in range(max_degree):
        for L in _enumerate_monomials(variables - 1, max_degree):
            t.append([i] + L)
    return t


def _is_modulus_condition(expression):
    if isinstance(expression, BinopExpr) \
            and expression.operator == Binop.EQ \
            and isinstance(expression.rhs, NatLitExpr) \
            and isinstance(expression.lhs, BinopExpr) \
            and expression.lhs.operator == Binop.MODULO:
        mod_expr = expression.lhs
        if isinstance(mod_expr.lhs, VarExpr) and isinstance(mod_expr.rhs, NatLitExpr):
            return True


def _term_generator(function: sympy.Poly):
    assert isinstance(function, sympy.Poly), "Terms can only be generated for finite GF"
    poly = function
    while poly.as_expr() != 0:
        yield sympy.S(poly.LC() * poly.LM().as_expr())
        poly -= poly.LC() * poly.LM().as_expr()


class GeneratingFunction:
    """
    This class represents a generating function. It wraps the sympy library.
    A GeneratingFunction object is designed to be immutable.
    This class does not ensure to be in a healthy state (i.e., every coefficient is non-negative).
    """
    rational_preciseness = False
    verbose_mode = False

    def __init__(self,
                 function: str = "",
                 variables: set = None,
                 preciseness=1.0,
                 closed: bool = None,
                 finite: bool = None):

        self._function = sympy.S(function, rational=True)
        self._variables = self._function.free_symbols
        self._preciseness = sympy.S(str(preciseness), rational=True)
        if variables:
            for variable in variables:
                self._variables = self._variables.union({sympy.S(variable)})
        self._dimension = len(self._variables)
        self._is_closed_form = closed if closed else not self._function.is_polynomial()
        self._is_finite = finite if finite else self._function.expand().ratsimp().is_polynomial()

    def _arithmetic(self, other, op: operator):
        if isinstance(other, GeneratingFunction):

            s, o = self.coefficient_sum(), other.coefficient_sum()
            is_closed_form = self._is_closed_form and other._is_closed_form
            is_finite = self._is_finite and other._is_finite
            preciseness = (s + o) / (s / self._preciseness + o / other._preciseness)
            function = op(self._function, other._function)
            variables = self._variables.union(other._variables)
            return GeneratingFunction(function, variables, preciseness, is_closed_form, is_finite)
        else:
            raise SyntaxError(f"You cannot {str(op)} {type(self)} with {type(other)}.")

    def __add__(self, other):
        return self._arithmetic(other, operator.add)

    def __sub__(self, other):
        return self._arithmetic(other, operator.sub)

    def __mul__(self, other):
        return self._arithmetic(other, operator.mul)

    def __truediv__(self, other):
        return self._arithmetic(other, operator.truediv)

    def __str__(self):
        if GeneratingFunction.rational_preciseness:
            output = f"{str(self._function)} \t@{str(self._preciseness)}"
            if GeneratingFunction.verbose_mode:
                output += f"\t({str(self._is_closed_form)},{str(self._is_finite)})"
            return output
        else:
            output = str(self._function) + "\t@" + str(self._preciseness.evalf())
            if GeneratingFunction.verbose_mode:
                output += f"\t({str(self._is_closed_form)},{str(self._is_finite)})"
            return output

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
            if self._is_finite:
                func = self._function.expand() if self._is_closed_form else self._function
                terms = func.as_coefficients_dict()
                for monomial in terms:
                    variables = monomial.as_powers_dict()
                    for var in variables:
                        if terms[monomial] >= other.probability_of({var: variables[var]}):
                            return False
                return True
            elif other._is_finite:
                func = other._function.expand() if other._is_closed_form else other._function
                terms = func.as_coefficients_dict()
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
        return not (self == other)

    def _monomial_to_state(self, monomial):
        result = dict()
        if monomial.free_symbols == set():
            for var in self._variables:
                result[var] = 0
        else:
            variables_and_powers = monomial.as_powers_dict()
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
        """
        This method assumes that the addend is given in terms of :math:`\alpha \times X^{\sigma}`, where
        :math:`\alpha \in [0,1], \sigma \in \mathbb{N}^k`.
        :param addend: the addend to split into its factor and monomial.
        :return: a tuple (factor, monomial)
        """
        if addend.free_symbols == set():
            return addend, sympy.S(1)
        else:
            factor_powers = addend.as_powers_dict()
            result = (sympy.S(1), sympy.S(1))
            for factor in factor_powers:
                if factor in addend.free_symbols:
                    result = (result[0], result[1] * factor ** factor_powers[factor])
                else:
                    result = (result[0] * factor ** factor_powers[factor], result[1])
            return result

    def diff(self, variable, k):
        """
        Partial `k`-th derivative of the generating function with respect to variable `variable`.
        :param variable: The variable in which the generating function gets differentiated.
        :param k: The order of the partial derivative.
        :return: The `k`-th partial derivative of the generating function in `variable`

        .. math:: \fraction{\delta G^`k`}{\delta `var`^`k`}
        """
        return GeneratingFunction(sympy.diff(self._function, sympy.S(variable), k), variables=self._variables,
                                  preciseness=self._preciseness)

    def _mult_term_generator(self):
        i = 1
        old_monomials = []
        while True:
            new_monomials = _enumerate_monomials(len(self._variables), i)
            for monomial in [item for item in new_monomials if item not in old_monomials]:
                index = 0
                func = self._function
                for var in self._variables:
                    func = func.diff(var, monomial[index]) / sympy.factorial(monomial[index])
                    func = func.limit(var, 0, "-") * var ** monomial[index]
                    index += 1
                yield func
            old_monomials = new_monomials
            logger.info(f"\t>Terms generated until total degree of {i - 1}")
            i += 1

    def as_series(self):
        if self._is_finite:
            if self._is_closed_form:
                func = self._function.expand().ratsimp().as_poly(list(self._variables))
            else:
                func = self._function.as_poly(gens=list(self._variables))
            return _term_generator(func)

        else:
            if 0 <= self._dimension <= 1:
                series = self._function.lseries()
                if str(type(series)) == "<class 'generator'>":
                    return series
                else:
                    return {self._function}
            else:
                logger.info("Multivariate Taylor expansion might take a while...")
                return self._mult_term_generator()

    def expand_until(self, threshold=None, nterms=None):
        approx = sympy.S("0")
        prec = sympy.S(0)
        if threshold:
            assert sympy.S(threshold) < self.coefficient_sum(), \
                f"Threshold cannot be larger than total coefficient sum! Threshold:" \
                f" {sympy.S(threshold)}, CSum {self.coefficient_sum()}"
            for term in self.as_series():
                if prec >= threshold:
                    break
                approx += term
                prec += self.split_addend(term)[0]
        else:
            assert nterms > 0, "Expanding to less than 0 terms is not valid."
            n = 0
            for term in self.as_series():
                if n >= nterms:
                    break
                approx += term
                prec += self.split_addend(term)[0]
        return GeneratingFunction(str(approx), self._variables, preciseness=prec, closed=False, finite=True)

    def dim(self):
        return self._dimension

    def coefficient_sum(self):
        coefficient_sum = self._function
        for var in self._variables:
            coefficient_sum = coefficient_sum.limit(var, 1, "-")
        return sympy.S(coefficient_sum)

    def vars(self):
        return self._variables

    def expected_value_of(self, variable: str):
        result = sympy.diff(self._function, sympy.S(variable))
        for var in self._variables:
            result = sympy.limit(result, sympy.S(var), 1, '-')
        return result

    def probability_of(self, state: dict):
        """
        Determines the probability of a single program state encoded by a monomial (discrete only).
        :param state: The queried program state.
        :return: The probability for that state.
        """

        if self._is_closed_form or not self._is_finite:
            result = self._function
            for variable, value in state.items():
                result = sympy.diff(result, sympy.S(variable), value)
                result /= sympy.factorial(value)
            for var in self.vars():
                result = result.subs(var, 0)
            return result
        else:
            monomial = sympy.S(1)
            for variable, value in state.items():
                monomial *= sympy.S(f"{variable} ** {value}")
            probability = self._function.as_poly().coeff_monomial(monomial)
            return probability if probability else sympy.core.numbers.Zero

    def normalized(self):
        mass = self.coefficient_sum()
        if mass == 0:
            raise ZeroDivisionError
        return GeneratingFunction(self._function / mass,
                                  variables=self._variables,
                                  preciseness=self._preciseness,
                                  closed=self._is_closed_form,
                                  finite=self._is_finite)

    def is_finite(self):
        """
        Checks whether the generating function is finite.
        :return: True if the GF is a polynomial, False otherwise.
        """
        return self._is_finite

    def simplify(self):
        return GeneratingFunction(self._function.simplify(),
                                  self._variables,
                                  self._preciseness,
                                  self._is_closed_form,
                                  self._is_finite)

    def evaluate(self, expression, monomial):
        op = expression.operator
        if isinstance(expression, UnopExpr):
            if op == Unop.NEG:
                return not self.evaluate(expression.expr, monomial)
            else:
                raise NotImplementedError(
                    "Iverson brackets are not supported.")
        elif isinstance(expression, BinopExpr):

            lhs = expression.lhs
            rhs = expression.rhs

            if op == Binop.AND:
                return self.evaluate(lhs, monomial) and self.evaluate(rhs, monomial)
            elif op == Binop.OR:
                return self.evaluate(lhs, monomial) or self.evaluate(rhs, monomial)
            elif op == Binop.EQ or op == Binop.LEQ or op == Binop.LE:
                if op == Binop.EQ:
                    equation = sympy.S(f"{lhs} - {rhs}")
                else:
                    equation = sympy.S(str(expression))
                variable_valuations = monomial.as_powers_dict()
                for var in self._variables:
                    if var not in variable_valuations.keys():
                        equation = equation.subs(var, 0)
                    else:
                        equation = equation.subs(var, variable_valuations[var])
                if op == Binop.EQ:
                    equation = equation == 0
                return equation
            else:
                raise AssertionError("Expression must be an (in-)equation!")
        else:
            raise AssertionError(
                "Expression has an unknown format and/or type.")

    def filter(self, expression: Expr):
        """
        Rough implementation of a filter. Can only handle distributions with finite support, or constant constraints on
        infinite support distributions.
        :return:
        """

        # Logical operators
        if expression.operator == Unop.NEG:
            result = self - self.filter(expression.expr)
            return result
        elif expression.operator == Binop.AND:
            result = self.filter(expression.lhs)
            return result.filter(expression.rhs)
        elif expression.operator == Binop.OR:
            neg_lhs = UnopExpr(operator=Unop.NEG, expr=expression.lhs)
            neg_rhs = UnopExpr(operator=Unop.NEG, expr=expression.rhs)
            conj = BinopExpr(operator=Binop.AND, lhs=neg_lhs, rhs=neg_rhs)
            neg_expr = UnopExpr(operator=Unop.NEG, expr=conj)
            return self.filter(neg_expr)

        # Modulo extractions
        elif _is_modulus_condition(expression):
            return self.arithmetic_progression(str(expression.lhs.lhs), str(expression.lhs.rhs))[expression.rhs.value]
        # Constant expressions
        elif _is_constant_constraint(expression):
            if expression.operator == Binop.LE:
                variable = sympy.S(str(expression.lhs))
                constant = expression.rhs.value
                result = sympy.S(0)
                for i in range(0, constant):
                    result += ((sympy.diff(self._function, variable, i) / sympy.factorial(i)).limit(
                        variable, 0) * variable ** i).simplify()
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

                new_func = self._function.diff(variable, constant)
                new_func /= sympy.factorial(constant)
                new_func = new_func.limit(variable, 0) * variable ** constant
                return GeneratingFunction(new_func, variables=self._variables, preciseness=self._preciseness)
            else:
                raise Exception(f"Expression is neither an equality nor inequality!")
        # evaluate on finite
        elif self._is_finite:
            result = sympy.S(0)
            addends = self._function.as_coefficients_dict() if not self._is_closed_form \
                else self._function.factor().expand().as_coefficients_dict()
            for monomial in addends:
                if self.evaluate(expression, monomial):
                    result += addends[monomial] * monomial
            return GeneratingFunction(result,
                                      self._variables,
                                      self._preciseness,
                                      closed=False,
                                      finite=True)
        else:
            raise NotComputable("Instruction {} is not computable on infinite generating function {}"
                                .format(expression, self._function))

    def limit(self, variable: str, value: str):
        return GeneratingFunction(self._function.limit(sympy.S(variable), sympy.S(value), "-"),
                                  preciseness=self._preciseness, closed=self._is_closed_form)

    def linear_transformation(self, variable, expression):
        # Transform expression into sympy readable format
        rhs = sympy.S(str(expression))
        subst_var = sympy.S(str(variable))

        # Check whether the expression contains the substitution variable or not
        terms = rhs.as_coefficients_dict()
        result = self._function
        if subst_var not in terms.keys():
            result = result.limit(subst_var, 1, '-') if self._is_closed_form else result.subs(subst_var, 1)

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
            # otherwise always assume we do an addition
            else:
                replacements.append((var, var * subst_var ** terms[var]))
        return GeneratingFunction(result.subs(replacements) * const_correction_term, variables=self._variables,
                                  preciseness=self._preciseness, closed=self._is_closed_form, finite=self._is_finite)

    def arithmetic_progression(self, variable: str, modulus: str):
        a = sympy.S(modulus)
        var = sympy.S(variable)
        primitive_uroot = sympy.S(f"exp(2 * {sympy.pi} * {sympy.I}/{a})")
        result = []
        for remainder in range(a):
            psum = 0
            for m in range(a):
                psum += primitive_uroot ** (-m * remainder) * self._function.subs(var, (primitive_uroot ** m) * var)
            result.append(GeneratingFunction(f"(1/{a}) * ({psum})", self._variables, self._preciseness,
                                             self._is_closed_form, self._is_finite))
        return result

    def _create_histogram_for_variable(self, var: str, n: int, p: str):
        marginal = self
        for variable in self._variables.difference({sympy.S(var)}):
            marginal = marginal.limit(variable, "1")
        if marginal._is_finite:
            data = []
            ind = []
            for addend in marginal.as_series():
                (prob, mon) = self.split_addend(addend)
                state = self._monomial_to_state(mon)
                data.append(prob)
                ind.append(float(state[sympy.S(var)]))
            ax = plt.subplot()
            ax.bar(ind, data, 1, linewidth=.5, ec=(0, 0, 0))
            ax.set_xlabel(f"{var}")
            ax.set_xticks(ind)
            ax.set_ylabel(f'Probability p({var})')
            plt.get_current_fig_manager().set_window_title("Histogram Plot")
            plt.gcf().suptitle("Histogram")
            plt.show()
        else:
            gf = marginal.expand_until(p, n)
            gf._create_histogram_for_variable(var, n, p)

    def _create_2d_hist(self, n, p):
        if self._is_finite:
            coord_and_prob = dict()
            maxima = dict()
            var_x = None
            var_y = None
            max_prob = 0
            for addend in self.as_series():
                (prob, mon) = self.split_addend(addend)
                state = self._monomial_to_state(mon)
                coord = ()
                i = 0
                for var in self._variables:
                    if i == 0:
                        var_x = var
                    elif i == 1:
                        var_y = var
                    if var not in maxima:
                        maxima[var] = state[var]
                    else:
                        maxima[var] = max(maxima[var], state[var])
                    coord += (state[var],)
                    i += 1
                coord_and_prob[coord] = prob
                max_prob = max(prob, max_prob)
            colors = []
            for _ in range(maxima[var_y] + 1):
                colors.append(list(0.0 for _ in range(maxima[var_x] + 1)))
            for coord in coord_and_prob:
                x = coord[0]
                y = coord[1]
                colors[y][x] = float(coord_and_prob[coord])
            c = plt.pcolormesh(colors, vmin=0, vmax=max_prob, shading='auto', ec="k", cmap="Blues", linewidth=0.5)
            plt.colorbar(c)
            plt.gca().set_xlabel(f"{var_x}")
            plt.gca().set_xticks(range(0, maxima[var_x] + 1))
            plt.gca().set_ylabel(f"{var_y}")
            plt.gca().set_yticks(range(0, maxima[var_y] + 1))
            plt.show()
        else:
            gf = self.expand_until(p, n)
            gf._create_2d_hist(n, p)

    def create_histogram(self, n=None, p: str = None, var: str = None):
        """
        Shows the histogram of the marginal distribution of the specified variable.
        """
        if var:
            self._create_histogram_for_variable(var, n, p)
        else:
            if len(self._function.free_symbols) > 2:
                raise Exception("Multivariate distributions need to specify the variable to plot")

            elif len(self._function.free_symbols) == 2:
                self._create_2d_hist(n, p)
            else:
                for var in self._function.free_symbols:
                    self._create_histogram_for_variable(str(var), n, p)
