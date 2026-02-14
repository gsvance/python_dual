"""module docstring"""

import math
import numbers
import operator
import re


__all__ = ['Dual']  # ep, eps, inf, infep, infeps, etc.


# ep or eps?
EPSILON = 'ep'


_DUAL_FORMAT = re.compile(r"""...""")


# Helper functions here
# - helpers for creating format strings...


class Dual(numbers.Number):
    """class docstring"""

    __slots__ = ('_real', '_dual')

    def __new__(cls, real=0.0, dual=None):
        # real could be a float, or a string we need to parse
        self = super(Dual, cls).__new__(cls)
        self._real = float(real)
        self._dual = float(dual) if dual is not None else 0.0
        return self

    # Alternate constructors as classmethods...
    # including constructors for internal use only

    # Return tuple of two floats

    @property
    def real(self):
        return self._real

    @property
    def dual(self):
        return self._dual

    def __repr__(self):
        return f"{self.__class__.__name__}({self._real!r}, {self._dual!r})"

    def __str__(self):
        if self._dual == 0.0:
            return str(self._real)
        if self._real == 0.0:
            return f"{self._dual!s}{EPSILON}"
        dual_sign = '+' if self._dual > 0.0 else '-'
        return f"{self._real!s}{dual_sign}{abs(self._dual)!s}{EPSILON}"

    # Format methods...

    # Operator fallbacks

    # Details for arithmetic algos

    # Adding

    # Subtracting

    # Multiplying

    # True dividing

    # Floordiv?

    # Divmod?

    # Mod?

    # Pow

    def __pos__(a):
        return Dual(a._real, a._dual)

    def __neg__(a):
        return Dual(-a._real, -a._dual)

    def __abs__(a):
        return math.hypot(a._real, a._dual)

    def conjugate(a):
        return Dual(a._real, -a._dual)

    def __int__(a):
        return int(a._real)

    # Trunc, Floor, Ceil, Round

    def __float__(a):
        return a._real

    def __complex__(a):
        return complex(a._real)

    def __hash__(self):
        if self._dual == 0.0:
            return hash(self._real)
        return hash((self._real, self._dual))

    # Eq and other comparisons
    # Does dual need <, >, <=, >= ?
    # Should they be compared pairwise like (real, dual) where epsilon gets
    # 2nd priority since it's an infinitesimal?

    def __bool__(a):
        return a._real != 0.0 or a._dual != 0.0

    # Support for pickling, copy, and deepcopy

    def __reduce__(self):
        return self.__class__, (self._real, self._dual)

    def __copy__(self):
        if type(self) == Dual:
            return self
        return self.__class__(self._real, self._dual)

    def __deepcopy__(self):
        if type(self) == Dual:
            return self
        return self.__class__(self._real, self._dual)


# ep or eps?
ep = Dual(0.0, 1.0)
eps = Dual(0.0, 1.0)

# Extensions of math functions to handle dual
# like sin and sqrt and such (see cmath)

inf = float('inf')
infep = Dual(0.0, inf)
infeps = Dual(0.0, inf)

nan = float('nan')
nanep = Dual(0.0, nan)
naneps = Dual(0.0, nan)
