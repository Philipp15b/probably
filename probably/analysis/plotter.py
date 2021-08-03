import logging
from typing import Union, Optional

import sympy
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from probably.analysis.exceptions import ParameterError
from probably.analysis.generating_function import GeneratingFunction
from probably.util.logger import log_setup

logger = log_setup(__name__, logging.DEBUG)


class Plotter:
    """ Plotter that generates histogram plots using matplotlib."""

    @staticmethod
    def _create_2d_hist(function: GeneratingFunction, var_1: sympy.Symbol, var_2: sympy.Symbol, n: Optional[int], p: Optional[sympy.Expr]):

        x = var_1
        y = var_2

        # Marginalize distribution to the variables of interest.
        marginal = function.marginal(var_1, var_2)
        marginal._variables = function.get_variables()
        logger.debug(f"Creating Histogram for {marginal}")
        # Collect relevant data from the distribution and plot it.
        if marginal.is_finite():
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
                state = marginal.monomial_to_state(mon)
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
            c = plt.imshow(colors, vmin=0, origin='lower', interpolation='nearest', cmap="turbo", aspect='auto')
            plt.colorbar(c)
            plt.gca().set_xlabel(f"{x}")
            plt.gca().set_xticks(range(0, maxima[x] + 1))
            plt.gca().set_ylabel(f"{y}")
            plt.gca().set_yticks(range(0, maxima[y] + 1))
            plt.show()
        else:
            # make the marginal finite.
            plt.ion()
            for subsum in marginal.expand_until(sympy.S(p), sympy.S(n)):
                Plotter._create_2d_hist(subsum, var_1, var_2, n, p)

    @staticmethod
    def _create_histogram_for_variable(function: GeneratingFunction, var: sympy.Symbol, n: sympy.Expr, p: sympy.Expr) -> None:
        marginal = function.marginal(var)
        if marginal.is_finite():
            data = []
            ind = []
            terms = prob_sum = 0
            for addend in marginal.as_series():
                if n and terms >= n:
                    break
                if p and prob_sum > p:
                    break
                (prob, mon) = GeneratingFunction.split_addend(addend)
                state = function.monomial_to_state(mon)
                data.append(float(prob))
                ind.append(float(state[var]))
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
                Plotter._create_histogram_for_variable(gf, var, n, p)

    @staticmethod
    def plot(function: GeneratingFunction, *variables: Union[str, sympy.Symbol], n: int = None, p: str = None) -> None:
        """ Shows the histogram of the marginal distribution of the specified variable(s). """

        probability = sympy.S(p) if p else None
        iterations = sympy.S(n) if n else None
        if variables:
            if len(variables) > 2:
                raise ParameterError(f"create_plot() cannot handle more than two variables!")
            if len(variables) == 2:
                Plotter._create_2d_hist(function, var_1=sympy.S(variables[0]), var_2=sympy.S(variables[1]), n=iterations, p=probability)
            if len(variables) == 1:
                Plotter._create_histogram_for_variable(function, var=sympy.S(variables[0]), n=iterations, p=probability)
        else:
            if len(function.get_variables()) > 2:
                raise Exception("Multivariate distributions need to specify the variable to plot")

            elif len(function.get_variables()) == 2:
                vars = list(function.get_variables())
                Plotter._create_2d_hist(function, var_1=vars[0], var_2=vars[1], n=iterations, p=probability)
            else:
                for var in function.get_variables():
                    Plotter._create_histogram_for_variable(var, n=iterations, p=probability)
