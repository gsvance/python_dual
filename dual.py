"""module docstring"""

import fractions
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

    def _operator_fallbacks(monomorphic_operator, fallback_operator):

        # This function doesn't actually use the fallback_operator at the
        # moment, so maybe that isn't needed?

        def forward(a, b):
            if isinstance(b, Dual):
                return monomorphic_operator(a, b)
            if isinstance(b, (int, float, fractions.Fraction)):
                return monomorphic_operator(a, Dual(b))
            # If b is complex, that should maybe be an error
            # There's no sensible way to compute, e.g., 1+2j + 3+4ep
            # At least, not without a hybrid class of some sort
            if isinstance(b, complex):
                pass
            return NotImplemented
        forward.__name__ = '__' + fallback_operator.__name__ + '__'
        forward.__doc__ = monomorphic_operator.__doc__

        def reverse(b, a):
            if isinstance(b, numbers.Real):
                return monomorphic_operator(Dual(float(a)), b)
            # Again, if b in complex, that should maybe be an error
            if isinstance(b, numbers.Complex):
                pass
            return NotImplemented
        reverse.__name__ = '__r' + fallback_operator.__name__ + '__'
        reverse.__doc__ = monomorphic_operator.__doc__

        return forward, reverse

    # Dual number arithmetic algorithms
    #
    # Dual numbers can be added component-wise, and multiplied by the formula
    #
    #   (a + b*ep) * (c + d*ep) = a*c + (a*d+b*c)*ep
    #
    # which follows from the property ep**2 == 0 and the fact that
    # multiplication is a bilinear operation.

    def _add(a, b):
        """a + b"""
        return Dual(a._real + b._real, a._dual + b._dual)

    __add__, __radd__ = _operator_fallbacks(_add, operator.add)

    def _sub(a, b):
        """a - b"""
        return Dual(a._real - b._real, a._dual - b._dual)

    __sub__, __rsub__ = _operator_fallbacks(_sub, operator.sub)

    def _mul(a, b):
        """a * b"""
        r = a._real * b._real
        d = a._real * b._dual + a._dual * b._real
        return Dual(r, d)

    __mul__, __rmul__ = _operator_fallbacks(_mul, operator.mul)

    def _div(a, b):
        """a / b"""
        if b._real == 0.0:
            raise ZeroDivisionError(
                f"division by dual number {b!s} with zero real part"
            )
        r = a._real / b._real
        d = a._dual / b._real - a._real * b._dual / b._real ** 2
        return Dual(r, d)

    __truediv__, __rtruediv__ = _operator_fallbacks(_div, operator.truediv)

    # Does it make any sense to implement floordiv, divmod, or mod? What would
    # they look like for dual numbers?

    # Pow

    def __pos__(a):
        return Dual(a._real, a._dual)

    def __neg__(a):
        return Dual(-a._real, -a._dual)

    def __abs__(a):
        return abs(self._real)

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

inf = float('inf')
infep = Dual(0.0, inf)
infeps = Dual(0.0, inf)

nan = float('nan')
nanep = Dual(0.0, nan)
naneps = Dual(0.0, nan)

# Extensions of math functions to handle dual
# like sin and sqrt and such (see cmath)

def abs2(x):
    """abs(x) ** 2"""
    return x.real ** 2
