import functools
import logging
from typing import Tuple, List, Set, Dict, Union

import sympy
import matplotlib.pyplot as plt
import operator
from matplotlib.cm import ScalarMappable
from probably.pgcl import Unop, VarExpr, NatLitExpr, BinopExpr, Binop, Expr, BoolLitExpr
from .exceptions import ComparisonException, NotComputable, ParameterError

logger = logging.getLogger("probably.analysis.generating_function")
logger.setLevel(logging.DEBUG)
fhandler = logging.FileHandler(filename='test.log', mode='w')
fhandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(fhandler)


def _is_constant_constraint(expression):  # Move this to expression checks etc.
    if isinstance(expression.lhs, VarExpr):
        if isinstance(expression.rhs, NatLitExpr):
            return True
    else:
        return False


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
        yield sympy.S(poly.EC() * poly.EM().as_expr())
        poly -= poly.EC() * poly.EM().as_expr()


class GeneratingFunction:
    """
    This class represents a generating function. It wraps the sympy library.
    A GeneratingFunction object is designed to be immutable.
    This class does not ensure to be in a healthy state (i.e., every coefficient is non-negative).
    """
    rational_preciseness = False
    verbose_mode = False
    simplified_output = False

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

    def __init__(self,
                 function: str = "",
                 variables: set = None,
                 preciseness=1.0,
                 closed: bool = None,
                 finite: bool = None):

        self._function: sympy.Expr = sympy.S(function, rational=True)
        self._variables: Set[sympy.Symbol] = self._function.free_symbols
        self._preciseness = sympy.S(str(preciseness), rational=True)
        if variables:
            for variable in variables:
                self._variables = self._variables.union({sympy.S(variable)})
        self._is_closed_form = closed if closed else not self._function.is_polynomial()
        self._is_finite = finite if finite else self._function.ratsimp().is_polynomial()

    def copy(self) -> 'GeneratingFunction':
        return GeneratingFunction(str(self._function), self._variables, self._preciseness, self._is_closed_form,
                                  self._is_finite)

    def _arithmetic(self, other, op: operator):
        if isinstance(other, GeneratingFunction):

            s, o = self.coefficient_sum(), other.coefficient_sum()
            is_closed_form = self._is_closed_form and other._is_closed_form
            is_finite = self._is_finite and other._is_finite
            preciseness = (s + o) / (s / self._preciseness + o / other._preciseness)
            function = op(self._function, other._function)
            logger.debug(f"Try simplification but stay in closed form!")
            n, d = sympy.fraction(function)
            logger.info("Factoring")
            n = n.factor()
            d = d.factor()
            function = n / d
            logger.debug(f"Closed-form simplification result: {function}")
            if not is_closed_form:
                logger.debug("Try to cancel terms")
                function = function.expand()
                logger.debug(f"Canceling result: {function}")
            variables = self._variables.union(other._variables)
            return GeneratingFunction(function, variables, preciseness, is_closed_form, is_finite)
        else:
            raise SyntaxError(f"You cannot {str(op)} {type(self)} with {type(other)}.")

    def __add__(self, other):
        return self._arithmetic(other, operator.add)

    def __sub__(self, other):
        logger.debug(f"Subtraction of {other} from {self}")
        return self._arithmetic(other, operator.sub)

    def __mul__(self, other):
        return self._arithmetic(other, operator.mul)

    def __truediv__(self, other):
        return self._arithmetic(other, operator.truediv)

    def __str__(self):
        if GeneratingFunction.rational_preciseness:
            output = f"{str(self._function.simplify() if GeneratingFunction.simplified_output else self._function)}" \
                     f" \t@{str(self._preciseness)}"
            if GeneratingFunction.verbose_mode:
                output += f"\t({str(self._is_closed_form)},{str(self._is_finite)})"
            return output
        else:
            output = str(self._function.simplify() if GeneratingFunction.simplified_output else self._function) + "\t@"
            output += str(self._preciseness.evalf())
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

    def _monomial_to_state(self, monomial) -> Dict[sympy.Expr, int]:
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

    def precision(self):
        return self._preciseness

    def diff(self, variable, k):
        """
        Partial `k`-th derivative of the generating function with respect to variable `variable`.
        :param variable: The variable in which the generating function gets differentiated.
        :param k: The order of the partial derivative.
        :return: The `k`-th partial derivative of the generating function in `variable`

        .. math:: \fraction{\delta G^`k`}{\delta `var`^`k`}
        """
        logger.debug(f"diff Call")
        return GeneratingFunction(sympy.diff(self._function, sympy.S(variable), k), variables=self._variables,
                                  preciseness=self._preciseness)

    def _mult_term_generator(self):
        i = 1
        while True:
            logger.debug(f"generating and sorting of new monomials")
            new_monomials = sorted(sympy.polys.monomials.itermonomials(self._variables, i),
                                   key=sympy.polys.orderings.monomial_key("grlex", list(self._variables)))
            if i > 1:
                new_monomials = new_monomials[sympy.polys.monomials.monomial_count(len(self._variables), i - 1):]
            logger.debug(f"Monomial_generation done")
            for monomial in new_monomials:
                logger.debug(f"create term for monomial {monomial}")
                mon_expr = monomial.as_expr()
                term = mon_expr * self.probability_of(mon_expr.as_powers_dict())
                logger.debug(f"created term {term}")
                yield term
            logger.debug(f"\t>Terms generated until total degree of {i}")
            i += 1

    def as_series(self):
        logger.debug(f"as_series() call")
        if self._is_finite:
            if self._is_closed_form:
                func = self._function.expand().ratsimp().as_poly(list(self._variables))
            else:
                func = self._function.as_poly(gens=list(self._variables))
            return _term_generator(func)

        else:
            if 0 <= len(self._variables) <= 1:
                series = self._function.lseries()
                if str(type(series)) == "<class 'generator'>":
                    return series
                else:
                    return {self._function}
            else:
                logger.debug("Multivariate Taylor expansion might take a while...")
                return self._mult_term_generator()

    def expand_until(self, threshold=None, nterms=None):
        logger.debug(f"expand_until() call")
        approx = sympy.S("0")
        prec = sympy.S(0)
        if threshold:
            assert sympy.S(threshold) < self.coefficient_sum(), \
                f"Threshold cannot be larger than total coefficient sum! Threshold:" \
                f" {sympy.S(threshold)}, CSum {self.coefficient_sum()}"
            for term in self.as_series():
                if prec >= sympy.S(threshold):
                    break
                approx += term
                prec += self.split_addend(term)[0]
                yield GeneratingFunction(str(approx.expand()), self._variables, preciseness=prec, closed=False,
                                         finite=True)
        else:
            assert nterms > 0, "Expanding to less than 0 terms is not valid."
            n = 0
            for term in self.as_series():
                if n >= nterms:
                    break
                approx += term
                prec += self.split_addend(term)[0]
                n += 1
                yield GeneratingFunction(str(approx.expand()), self._variables, preciseness=prec, closed=False,
                                         finite=True)

    def coefficient_sum(self):
        logger.debug(f"coefficient_sum() call")
        coefficient_sum = self._function.simplify()
        for var in self._variables:
            coefficient_sum = coefficient_sum.limit(var, 1, "-") if self._is_closed_form else coefficient_sum.subs(var,
                                                                                                                   1)
        return coefficient_sum

    def vars(self):
        return self._variables

    def expected_value_of(self, variable: str):
        logger.debug(f"expected_value_of() call")
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
        logger.debug(f"probability_of({state}) call")
        marginal = self.marginal([str(var) for var in self._variables.difference(state.keys())])
        if marginal._is_closed_form or not marginal._is_finite:
            result = self._function
            for variable, value in state.items():
                if not variable == 1:
                    result = result.diff(sympy.S(variable), value)
                    result /= sympy.factorial(value)
                    result = result.limit(variable, 0, "-") if marginal._is_closed_form else result.subs(variable, 0)
            return result
        else:
            monomial = sympy.S(1)
            for variable, value in state.items():
                monomial *= sympy.S(f"{variable} ** {value}")
            probability = marginal._function.as_poly(gens=marginal._variables).coeff_monomial(monomial)
            return probability if probability else sympy.core.numbers.Zero()

    def normalized(self):
        logger.debug(f"normalized() call")
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
        simplified = self._function.simplify()
        return GeneratingFunction(simplified,
                                  self._variables,
                                  self._preciseness,
                                  simplified.is_polynomial(),
                                  simplified.ratsimp().is_polynomial())

    @classmethod
    def evaluate(cls, expression: str, monomial: sympy.Expr) -> sympy.Expr:
        sexp = sympy.S(expression)
        for var, value in monomial.as_powers_dict().items():
            sexp = sexp.subs(var, value)
        for var in sexp.free_symbols:
            sexp = sexp.subs(var, 0)
        return sexp

    @classmethod
    def evaluate_condition(cls, condition: BinopExpr, monomial: sympy.Expr) -> bool:
        logger.debug(f"evaluate_condition() call")
        if not isinstance(condition, BinopExpr):
            raise AssertionError(f"Expression must be an (in-)equation, was {condition}")

        lhs = str(condition.lhs)
        rhs = str(condition.rhs)
        op = condition.operator

        if op == Binop.EQ:
            return GeneratingFunction.evaluate(lhs, monomial) == GeneratingFunction.evaluate(rhs, monomial)
        elif op == Binop.LEQ:
            return GeneratingFunction.evaluate(lhs, monomial) <= GeneratingFunction.evaluate(rhs, monomial)
        elif op == Binop.LE:
            return GeneratingFunction.evaluate(lhs, monomial) < GeneratingFunction.evaluate(rhs, monomial)
        raise AssertionError(f"Unexpected condition type. {condition}")

    def marginal(self, variables: List[str]) -> 'GeneratingFunction':
        """
        Computes the marginal distribution in the given variables.
        :param variables: a list of variables for which the marginal distribution should be computed
        :return: the marginal distribution.
        """
        logger.debug(f"Creating marginal for variables {variables} and joint probability distribution {self}")
        marginal = self.copy()
        for var in marginal._variables:
            if str(var) not in variables:
                marginal._function = marginal._function.limit(var, 1, "-") if marginal._is_closed_form else marginal._function.subs(var, 1)
                marginal._is_closed_form = not marginal._function.is_polynomial()
                marginal._is_finite = marginal._function.ratsimp().is_polynomial()
        marginal._variables = marginal._function.free_symbols
        return marginal

    def filter(self, expression: Expr) -> 'GeneratingFunction':
        """
        Rough implementation of a filter. Can only handle distributions with finite support, or constant constraints on
        infinite support distributions.
        :return:
        """
        logger.debug(f"filter({expression}) call on {self}")

        # Boolean literals
        if isinstance(expression, BoolLitExpr):
            return self
        # Logical operators
        if expression.operator == Unop.NEG:
            result = self - self.filter(expression.expr)
            return result
        elif expression.operator == Binop.AND:
            result = self.filter(expression.lhs)
            return result.filter(expression.rhs)
        elif expression.operator == Binop.OR:
            filtered = self.filter(expression.lhs)
            return filtered + self.filter(expression.rhs) - filtered.filter(expression.rhs)

        # Modulo extractions
        elif _is_modulus_condition(expression):
            return self.arithmetic_progression(str(expression.lhs.lhs), str(expression.lhs.rhs))[expression.rhs.value]
        # Constant expressions
        elif _is_constant_constraint(expression):
            variable = sympy.S(str(expression.lhs))
            constant = expression.rhs.value
            result = sympy.S(0)
            ranges = {Binop.LE: range(constant), Binop.LEQ: range(constant + 1), Binop.EQ: [constant]}

            for i in ranges[expression.operator]:
                tmp = self._function.expand().diff(variable, i) if not self._is_finite else self._function.diff(variable,i)
                tmp /= sympy.factorial(i)
                tmp = tmp.subs(variable, 0) if tmp.is_polynomial() else tmp.limit(variable, 0, '-')
                tmp *= variable ** i
                result += tmp
            return GeneratingFunction(result,
                                      variables=self._variables,
                                      preciseness=self._preciseness,
                                      closed=self._is_closed_form,
                                      finite=self._is_finite)
        elif self._is_finite:
            result = sympy.S(0)
            addends = self._function.as_coefficients_dict() if not self._is_closed_form \
                else self._function.cancel().expand().as_coefficients_dict()
            for monomial in addends:
                if self.evaluate_condition(expression, monomial):
                    result += addends[monomial] * monomial
            return GeneratingFunction(result,
                                      self._variables,
                                      self._preciseness,
                                      closed=False,
                                      finite=True)
        else:
            expr = sympy.S(str(expression.rhs))
            variables = [str(var) for var in expr.free_symbols]
            marginal = self.marginal(variables)
            left_side = True
            if not marginal.is_finite():
                left_side = False
                expr = sympy.S(str(expression.lhs))
                variables = [str(var) for var in expr.free_symbols]
                marginal = self.marginal(variables)
                if not marginal.is_finite():
                    raise NotComputable(f"Instruction {expression} is not computable on infinite generating function"
                                        f" {self._function}")
            else:
                print("generating terms")
                filter_exprs = []
                for term in marginal.as_series():
                    state_filter = []
                    tmp_expr = expr
                    prob, mon = GeneratingFunction.split_addend(term)
                    for var, val in self._monomial_to_state(mon).items():
                        if var in marginal._variables:
                            eq = BinopExpr(operator=Binop.EQ, lhs=VarExpr(str(var)), rhs=NatLitExpr(val))
                            tmp_expr = tmp_expr.subs(var, val)
                            state_filter.append(eq)
                    filter_expr = functools.reduce(
                        lambda right, left: BinopExpr(operator=Binop.AND, lhs=left, rhs=right),
                        state_filter,
                        BinopExpr(operator=expression.operator, lhs=expression.lhs, rhs=NatLitExpr(value=int(tmp_expr))) if left_side else
                        BinopExpr(operator=expression.operator, lhs=NatLitExpr(value=int(tmp_expr)), rhs=expression.rhs)
                    )
                    filter_exprs.append(filter_expr)

                new_expr = functools.reduce(lambda left, right: BinopExpr(operator=Binop.OR, lhs=left, rhs=right),
                                            filter_exprs[1:],
                                            filter_exprs[0]
                )
                return self.filter(new_expr)

    def limit(self, variable: Union[str, sympy.Symbol], value: str) -> 'GeneratingFunction':
        return GeneratingFunction(self._function.limit(sympy.S(variable), sympy.S(value), "-"),
                                  preciseness=self._preciseness, closed=self._is_closed_form)

    def linear_transformation(self, variable: str, expression: Expr) -> 'GeneratingFunction':
        logger.debug(f"linear_transformation() call")
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

    def arithmetic_progression(self, variable: str, modulus: str) -> List['GeneratingFunction']:
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
            prob_sum = 0
            terms = 0
            for addend in marginal.as_series():
                if n and terms >= sympy.S(n):
                    break
                if p and prob_sum > sympy.S(p):
                    break
                (prob, mon) = self.split_addend(addend)
                state = self._monomial_to_state(mon)
                data.append(float(prob))
                ind.append(float(state[sympy.S(var)]))
                prob_sum += prob
                terms += 1
            ax = plt.subplot()
            my_cmap = plt.cm.get_cmap("Blues")
            colors = my_cmap([x / max(data) for x in data])
            sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0, max(data)))
            sm.set_array([])
            ax.bar(ind, data, 1, linewidth=.5, ec=(0, 0, 0), color=colors)
            ax.set_xlabel(f"{var}")
            ax.set_xticks(ind)
            ax.set_ylabel(f'Probability p({var})')
            plt.get_current_fig_manager().set_window_title("Histogram Plot")
            plt.gcf().suptitle("Histogram")
            plt.colorbar(sm)
            plt.show()
        else:
            for gf in marginal.expand_until(p, n):
                gf._create_histogram_for_variable(var, n, p)

    def _create_2d_hist(self, var_1: str, var_2: str, n, p):

        x = sympy.S(var_1)
        y = sympy.S(var_2)

        # Marginalize distribution to the variables of interest.
        marginal = self._function
        for var in self._variables:
            if var != x and var != y:
                marginal = marginal.limit(var, 1, "-")
        marginal = GeneratingFunction(marginal.ratsimp())
        logger.debug(f"Creating Histogram for {marginal}")
        # Collect relevant data from the distribution and plot it.
        if marginal._is_finite:
            coord_and_prob = dict()
            maxima = {x: 0, y: 0}
            max_prob = 0
            colors = []

            # collect the coordinates and probabilities. Also compute maxima of probabilities and degrees
            terms = 0
            prob_sum = 0
            for addend in marginal.as_series():
                if p and prob_sum >= sympy.S(p):
                    break
                if n and terms >= sympy.S(n):
                    break
                (prob, mon) = marginal.split_addend(addend)
                state = marginal._monomial_to_state(mon)
                maxima[x], maxima[y] = max(maxima[x], state[x]), max(maxima[y], state[y])
                coord = (state[x], state[y])
                coord_and_prob[coord] = prob
                max_prob = max(prob, max_prob)
                terms += 1
                prob_sum += prob

            # Zero out the colors array
            for _ in range(maxima[y] + 1):
                colors.append(list(0.0 for _ in range(maxima[x] + 1)))

            # Fill the colors array with the previously collected data.
            for coord in coord_and_prob:
                colors[coord[1]][coord[0]] = float(coord_and_prob[coord])

            # Plot the colors array
            c = plt.imshow(colors, origin='lower', interpolation='nearest', cmap="turbo", aspect='auto')
            plt.colorbar(c)
            plt.gca().set_xlabel(f"{x}")
            plt.gca().set_xticks(range(0, maxima[x] + 1))
            plt.gca().set_ylabel(f"{y}")
            plt.gca().set_yticks(range(0, maxima[y] + 1))
            plt.show()
        else:
            # make the marginal finite.
            for marginal in marginal.expand_until(sympy.S(p), sympy.S(n)):
                if len(marginal._function.free_symbols) > 1:
                    marginal._create_2d_hist(var_1, var_2, n, p)

    def create_histogram(self, n=None, p: str = None, var: [str] = None):
        """
        Shows the histogram of the marginal distribution of the specified variable.
        """
        if var:
            if len(var) > 2:
                raise ParameterError(f"create_histogram() cannot handle more than two variables!")
            if len(var) == 2:
                self._create_2d_hist(var_1=var[0], var_2=var[1], n=n, p=p)
            if len(var) == 1:
                self._create_histogram_for_variable(var=var[0], n=n, p=p)
        else:
            if len(self._function.free_symbols) > 2:
                raise Exception("Multivariate distributions need to specify the variable to plot")

            elif len(self._function.free_symbols) == 2:
                vars = list(self._function.free_symbols)
                self._create_2d_hist(var_1=vars[0], var_2=vars[1], n=n, p=p)
            else:
                for var in self._function.free_symbols:
                    self._create_histogram_for_variable(str(var), n, p)

    def safe_filter(self, condition: Expr) -> Tuple['GeneratingFunction', 'GeneratingFunction', bool]:
        try:
            logger.info(f"filtering for {condition}")
            sat_part = self.filter(condition)
            non_sat_part = self - sat_part
            return sat_part, non_sat_part, False
        except NotComputable as err:
            print(err)
            probability = input("Continue with approximation. Enter a probability (0, {}):\t"
                                .format(self.coefficient_sum()))
            if sympy.S(probability) > sympy.S(0):
                approx = list(self.expand_until(probability))[-1]
                approx_sat_part = approx.filter(condition)
                approx_non_sat_part = approx - approx_sat_part
                return approx_sat_part, approx_non_sat_part, True
            else:
                raise NotComputable(str(err))
