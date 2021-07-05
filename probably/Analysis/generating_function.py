import sympy

class GeneratingFunction:

    """
    This class represents a generating function. It wraps the sympy library.
    """

    def __init__(self, variables: str = "", function: sympy.Function = 0):
        self._function = function
        self._variables = {}
        for var in str.split(variables):
            self._variables[var] = sympy.symbols(var)
        self._dimension = len(self._variables)
        sympy.init_printing()

    def __add__(self, other):
        self._function += other
        return self

    def __sub__(self, other):
        self._function -= other
        return self

    def __mul__(self, other):
        self._function *= other
        return self

    def __truediv__(self, other):
        self._function /= other
        return self

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

    def dim(self):
        return self._dimension

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
        for var in self._variables:
            result = result.subs(var, 0)
        return result
