"""module docstring"""

import abc
import decimal
import fractions
import math
import numbers
import operator
import re


__all__ = [
    'set_epsilon_variant',
    'Dual',
    'ep',
    'inf',
    'infep',
    'nan',
    'nanep',
    'abs2',
]


class AbstractDual(numbers.Number):
    """AbstractDual defines the operations that work on dual numbers.

    In the Python numerical hierarchy, AbstractDual exists "to the side of"
    Complex. While dual numbers are definitely a kind of Number and every Real
    is a valid dual number, they are not compatible with complex numbers. Dual
    numbers have a real part, no imaginary part, and a "dual part," and their
    "dual conjugate" operation flips the sign on the dual part rather than the
    imaginary part, with makes it inconsistent with the complex conjugate.
    """
    __slots__ = ()

    @abc.abstractmethod
    def as_dual(self):
        """Convert self to a concrete dual number with float parts."""
        raise NotImplementedError

    def __bool__(self):
        """Return True if self is nonzero and False otherwise."""
        return self != 0

    @property
    @abc.abstractmethod
    def real(self):
        """Return the real part of this dual number."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dual(self):
        """Return the dual part of this dual number."""
        raise NotImplementedError

    @property
    def imag(self):
        """Return the imaginary part of this dual number."""
        return 0

    @abc.abstractmethod
    def __add__(self, other):
        """self + other"""
        raise NotImplementedError

    @abc.abstractmethod
    def __radd__(self, other):
        """other + self"""
        raise NotImplementedError

    def __sub__(self, other):
        """self - other"""
        return self + -other

    def __rsub__(self, other):
        """other - self"""
        return other + -self

    @abc.abstractmethod
    def __mul__(self, other):
        """self * other"""
        raise NotImplementedError

    @abc.abstractmethod
    def __rmul__(self, other):
        """other * self"""
        raise NotImplementedError

    @abc.abstractmethod
    def __truediv__(self, other):
        """self / other"""
        raise NotImplementedError

    @abc.abstractmethod
    def __rtruediv__(self, other):
        """other / self"""
        raise NotImplementedError

    @abc.abstractmethod
    def __pow__(self, other):
        """self ** other"""
        raise NotImplementedError

    @abc.abstractmethod
    def __rpow__(self, other):
        """other ** self"""
        raise NotImplementedError

    @abc.abstractmethod
    def __pos__(self):
        """+self"""
        raise NotImplementedError

    @abc.abstractmethod
    def __neg__(self):
        """-self"""
        raise NotImplementedError

    @abc.abstractmethod
    def __abs__(self):
        """Returns the non-negative norm of a dual number."""
        raise NotImplementedError

    @abc.abstractmethod
    def conjugate(self):
        """Return self with the sign of its dual part flipped."""
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        """self == other"""
        raise NotImplementedError


# Note that this registration only affects isinstance() checks. The Real class
# does not inherit any methods implemented by AbstractDual as part of its MRO.
AbstractDual.register(numbers.Real)


# Since Real does not inherit methods from AbstractDual in its MRO, we define
# getter functions to handle extracting the real and dual parts from any kind
# of AbstractDual or Real in a uniform manner. These two helper functions will
# always return floats.


def _get_real_float(x: AbstractDual) -> float:
    """Retrieve the real part of any AbstractDual as a float."""
    if type(x) is float:
        return x
    if type(x) is Dual:
        return x.real
    if isinstance(x, numbers.Real):
        return float(x)
    if isinstance(x, AbstractDual):
        return float(x.real)
    raise TypeError(
        f"argument should be an AbstractDual, not {x!r} ({type(x).__name__})"
    )


def _get_dual_float(x: AbstractDual) -> float:
    """Retrieve the dual part of any AbstractDual as a float."""
    if type(x) is float:
        return 0.0
    if type(x) is Dual:
        return x.dual
    if isinstance(x, numbers.Real):
        return 0.0
    if isinstance(x, AbstractDual):
        return float(x.dual)
    raise TypeError(
        f"argument should be an AbstractDual, not {x!r} ({type(x).__name__})"
    )


# Several possible string representations for epsilon
_EPSILON_VARIANTS = {
    "ascii": "ep",
    "minuscule": chr(0x03B5),
    "lunate": chr(0x03F5),
    "latin": chr(0x025B),
}

# Default to the ASCII representation for epsilon
_epsilon = _EPSILON_VARIANTS["ascii"]


def set_epsilon_variant(v, /):
    """Set which representation of epsilon is used for output strings."""
    global _epsilon
    try:
        _epsilon = _EPSILON_VARIANTS[v.lower()]
    except KeyError as exc:
        raise ValueError(f"{v!r} is not a known epsilon variant") from exc


# A few component strings to use as building blocks for bigger regexes
_LEADING_WHITESPACE = r"\A\s*"
_FLOAT_WITH_OPTIONAL_SIGN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_FLOAT_WITH_MANDATORY_SIGN = r"[-+](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ANY_EPSILON_VARIANT = "(?:" + "|".join(_EPSILON_VARIANTS.values()) + ")"
_TRAILING_WHITESPACE = r"\s*\Z"

# Three regexes: real part only, dual part only, and both parts present
_DUAL_FORMAT_REAL_ONLY = re.compile(
    _LEADING_WHITESPACE
    + "(" + _FLOAT_WITH_OPTIONAL_SIGN + ")"
    + _TRAILING_WHITESPACE
)
_DUAL_FORMAT_DUAL_ONLY = re.compile(
    _LEADING_WHITESPACE
    + "(" + _FLOAT_WITH_OPTIONAL_SIGN + ")" + _ANY_EPSILON_VARIANT
    + _TRAILING_WHITESPACE
)
_DUAL_FORMAT_BOTH_PARTS = re.compile(
    _LEADING_WHITESPACE
    + "(" + _FLOAT_WITH_OPTIONAL_SIGN + ")"
    + "(" + _FLOAT_WITH_MANDATORY_SIGN + ")" + _ANY_EPSILON_VARIANT
    + _TRAILING_WHITESPACE
)


# Helper functions here
# - helpers for creating format strings...


class Dual(numbers.Number):
    """class docstring"""

    __slots__ = ('_real', '_dual')

    # Dual numbers are immutable, so use __new__ instead of __init__
    def __new__(cls, real=0.0, dual=None):
        """constructor docstring"""
        self = super(Dual, cls).__new__(cls)

        if dual is None:

            if type(real) is float:
                self._real = real
                self._dual = 0.0
                return self

            if isinstance(real, Dual):
                self._real = real.real
                self._dual = real.dual
                return self

            if (
                isinstance(real, numbers.Real)
                or (not isinstance(real, type) and hasattr(real, '__float__'))
            ):
                self._real = float(real)
                self._dual = 0.0
                return self

            if isinstance(real, str):
                # Handle parsing a dual number from a string
                match_both_parts = _DUAL_FORMAT_BOTH_PARTS.fullmatch(real)
                if match_both_parts is not None:
                    self._real = float(match_both_parts.group(1))
                    self._dual = float(match_both_parts.group(2))
                    return self
                match_dual_only = _DUAL_FORMAT_DUAL_ONLY.fullmatch(real)
                if match_dual_only is not None:
                    self._real = 0.0
                    self._dual = float(match_dual_only.group(1))
                    return self
                match_real_only = _DUAL_FORMAT_REAL_ONLY.fullmatch(real)
                if match_real_only is not None:
                    self._real = float(match_real_only.group(1))
                    self._dual = 0.0
                    return self
                raise ValueError(f"invalid literal for Dual: {real!r}")

            raise TypeError(
                "argument should be a string or a Dual "
                + "instance or be convertable to float"
            )

        if type(real) is float is type(dual):
            self._real = real
            self._dual = dual
            return self

        if isinstance(real, Dual) and isinstance(dual, Dual):
            self._real = real.real
            self._dual = dual.real + real.dual
            return self

        if isinstance(real, numbers.Real) and isinstance(dual, numbers.Real):
            self._real = float(real)
            self._dual = float(dual)
            return self

        if isinstance(real, Dual) and isinstance(dual, numbers.Real):
            self._real = real.real
            self._dual = float(dual)
            return self

        if isinstance(real, numbers.Real) and isinstance(dual, Dual):
            self._real = float(real)
            self._dual = dual.real
            return self

        raise TypeError("both arguments should be Dual or Real instances")

    @classmethod
    def from_number(cls, number):
        """Convert other types of number to a dual number instance."""
        if type(number) is float:
            return cls(number)
        if (
            isinstance(number, numbers.Real)
            or (not isinstance(number, type) and hasattr(number, '__float__'))
        ):
            return cls(float(number))
        raise TypeError("number should be a float or be convertable to float")

    @classmethod
    def from_float(cls, float_):
        """Convert a floating point number to a dual number instance."""
        if type(float_) is float:
            return cls(float_)
        raise TypeError(
            f"{cls.__name__}.from_float() accepts only floats"
            + f", not {float_!r} ({type(float_).__name__})"
        )

    @classmethod
    def from_decimal(cls, decimal_):
        """Create a dual number by converting a Decimal instance to a float."""
        if isinstance(decimal_, decimal.Decimal):
            return cls(float(decimal_))
        raise TypeError(
            f"{cls.__name__}.from_decimal() accepts only Decimal instances"
            + f", not {decimal_!r} ({type(decimal_).__name__})"
        )

    def as_float_pair(self):
        """Return a pair of floats (real, dual) composing the dual number."""
        return self._real, self._dual

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
            return f"{self._dual!s}{_epsilon}"
        dual_sign = '+' if self._dual > 0.0 else '-'
        return f"{self._real!s}{dual_sign}{abs(self._dual)!s}{_epsilon}"

    # Format methods...

    def _operator_fallbacks(monomorphic_operator, fallback_operator):

        # This function doesn't actually use the fallback_operator at the
        # moment, so maybe that isn't needed?

        def forward(a, b):
            if isinstance(b, Dual):
                return monomorphic_operator(a, b)
            if isinstance(b, (int, float, fractions.Fraction)):
                return monomorphic_operator(a, Dual(b))
            # If b is something else (including any subtype of Complex), give
            # the other object a chance to decide what happens. If it has no
            # ideas either, then a TypeError will be raised.
            return NotImplemented
        forward.__name__ = '__' + fallback_operator.__name__ + '__'
        forward.__doc__ = monomorphic_operator.__doc__

        def reverse(b, a):
            if isinstance(a, numbers.Real):
                return monomorphic_operator(Dual(float(a)), b)
            # If b is something else (including subtypes of Complex), then we
            # do not implement a solution. This return value will lead to a
            # TypeError being raised.
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
    # which follows from the property `ep**2 == 0` and the fact that
    # multiplication is a bilinear operation.
    #
    # To divide one dual number by another, we can take a similar approach as
    # when diving two complex numbers. We multiply both the numerator and
    # denominator by the conjugate of the denominator to remove the dual part
    # of the denominator. This leads to the formula
    #
    #   a + b*ep   a   (b*c-a*d)*ep
    #   -------- = - + ------------
    #   c + d*ep   c       c^2
    #
    # where `a/c`` is the real part of the quotient.
    #
    # In general, the operation of exponentiation is given by the formula
    #
    #   (a + b*ep) ** (c + d*ep) = a**c + (a**c)*(d*log(a)+(b*c)/a)*ep
    #
    # which simplifies considerably if any of the values of `b`, `c`, or `d`
    # are zero. However, it is not possible if `a` is zero given the required
    # log of and division by `a`.

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

    def _pow(a, b):
        """a ** b"""
        if a._real == 0.0:
            raise ZeroDivisionError(
                f"applying exponent to dual number {a!s} with zero real part"
            )
        r = a._real ** b._real
        d = r * (b._dual * math.log(a._real) + a._dual * b._real / a._real)
        return Dual(r, d)

    __pow__, __rpow__ = _operator_fallbacks(_pow, operator.pow)

    def __pos__(a):
        return Dual(a._real, a._dual)

    def __neg__(a):
        return Dual(-a._real, -a._dual)

    def __abs__(a):
        return abs(a._real)

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

    def __eq__(a, b):
        """a == b"""
        if type(b) is float or type(b) is int:
            return a._real == b and a._dual == 0.0
        if isinstance(b, Dual):
            return a._real == b.real and a._dual == b.dual
        if isinstance(b, numbers.Real):
            return a._real == float(b) and a._dual == 0.0
        if isinstance(b, numbers.Complex) and b.imag == 0.0:
            return a._real == b.real and a._dual == 0.0
        # Since a does not know how to compare with b, give b the chance to
        # compare itself with a
        return NotImplemented

    # Don't need to implement __ne__ since object.__ne__ just inverts __eq__

    def _richcmp(self, other, op):
        if isinstance(other, Dual):
            return op(self.as_float_pair(), other.as_float_pair())
        if isinstance(other, numbers.Real):
            return op(self.as_float_pair(), (float(other), 0.0))
        return NotImplemented

    def __lt__(a, b):
        """a < b"""
        return a._richcmp(b, operator.lt)

    def __gt__(a, b):
        """a > b"""
        return a._richcmp(b, operator.gt)

    def __le__(a, b):
        """a <= b"""
        return a._richcmp(b, operator.le)

    def __ge__(a, b):
        """a >= b"""
        return a._richcmp(b, operator.ge)

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


# Dual number mathematical constants


ep = Dual(0.0, 1.0)

inf = float('inf')
infep = Dual(0.0, inf)

nan = float('nan')
nanep = Dual(0.0, nan)

# Are these useful?
pi = math.pi
e = math.e
tau = math.tau


# Keep in mind for all these functions that
#   f(a + b*ep) = f(a) + b*f'(a)*ep
# for the dual numbers, where f' is the derivative of f


# Dual number coordinate conversions


def abs2(x):
    """abs(x) ** 2"""
    return x.real ** 2


def slope(x):
    """m such that x = r * (1 + m*ep)"""
    if x.real == 0.0:
        raise ZeroDivisionError(
            f"slope of dual number {x!s} with real part zero"
        )
    return x.dual / x.real


def polar(x):
    """(r, m) such that x = r * (1 + m*ep)"""
    return x.real, slope(x)


def rect(r, m):
    """r * (1 + m*ep)"""
    return Dual(r, r * m)


# Functions for exponents, logarithms, and roots


_LOG_2 = math.log(2.0)
_LOG_10 = math.log(10.0)


def exp(x):
    """exp(x) with dy/dx = exp(x)"""
    if isinstance(x, Dual):
        if x.real == 0.0:
            return Dual(1.0, x.dual)
        exp_real = math.exp(x.real)
        return Dual(exp_real, x.dual * exp_real)
    return Dual(math.exp(x))


def exp2(x):
    """exp2(x) with dy/dx = log(2)*exp2(x)"""
    if isinstance(x, Dual):
        if x.real == 0.0:
            return Dual(1.0, x.dual * _LOG_2)
        exp2_real = math.exp2(x.real)
        return Dual(exp2_real, x.dual * _LOG_2 * exp2_real)
    return Dual(math.exp2(x))


def expm1(x):
    """expm1(x) with dy/dx = exp(x)"""
    if isinstance(x, Dual):
        if x.real == 0.0:
            return Dual(0.0, x.dual)
        return Dual(math.expm1(x.real), x.dual * math.exp(x.real))
    return Dual(math.expm1(x))


def log(x):
    """log(x) with dy/dx = 1/x"""
    if isinstance(x, Dual):
        if x.real == 1.0:
            return Dual(0.0, x.dual)
        return Dual(math.log(x.real), x.dual / x.real)
    return Dual(math.log(x))


def log2(x):
    """log2(x) with dy/dx = 1/(x*log(2))"""
    if isinstance(x, Dual):
        if x.real == 1.0:
            return Dual(0.0, x.dual / _LOG_2)
        return Dual(math.log2(x.real), x.dual / (x.real * _LOG_2))
    return Dual(math.log2(x))


def log10(x):
    """log10(x) with dy/dx = 1/(x*log(10))"""
    if isinstance(x, Dual):
        if x.real == 1.0:
            return Dual(0.0, x.dual / _LOG_10)
        return Dual(math.log10(x.real), x.dual / (x.real * _LOG_10))
    return Dual(math.log10(x))


def log1p(x):
    """log1p(x) with dy/dx = 1/(1+x)"""
    if isinstance(x, Dual):
        if x.real == 0.0:
            return Dual(0.0, x.dual)
        return Dual(math.log1p(x.real), x.dual / (1.0 + x.real))
    return Dual(math.log1p(x))


def sqrt(x):
    """sqrt(x) with dy/dx = 1/(2*sqrt(x))"""
    if isinstance(x, Dual):
        if x.real == 1.0:
            return Dual(1.0, x.dual * 0.5)
        sqrt_real = math.sqrt(x.real)
        return Dual(sqrt_real, x.dual / (2.0 * sqrt_real))
    return Dual(math.sqrt(x))


def cbrt(x):
    """cbrt(x) with dydx = 1/(3*(cbrt(x))**2)"""
    if isinstance(x, Dual):
        if x.real == 1.0:
            return Dual(1.0, x.dual / 3.0)
        cbrt_real = math.cbrt(x.real)
        return Dual(cbrt_real, x.dual / (3.0 * cbrt_real**2))
    return Dual(math.cbrt(x))


# Trigonometric and inverse trigonometric functions


def degrees(x):
    """degrees(x) with dy/dx = 180/pi"""
    if isinstance(x, Dual):
        return Dual(math.degrees(x.real), math.degrees(x.dual))
    return Dual(math.degrees(x))


def radians(x):
    """radians(x) with dy/dx = pi/180"""
    if isinstance(x, Dual):
        return Dual(math.radians(x.real), math.radians(x.dual))
    return Dual(math.radians(x))


def sin(x):
    """sin(x) with dy/dx = cos(x)"""
    if isinstance(x, Dual):
        return Dual(math.sin(x.real), x.dual * math.cos(x.real))
    return Dual(math.sin(x))


def cos(x):
    """cos(x) with dy/dx = -sin(x)"""
    if isinstance(x, Dual):
        return Dual(math.cos(x), -x.dual * math.sin(x.real))
    return Dual(math.cos(x))


def tan(x):
    """tan(x) with dy/dx = (sec(x))**2"""
    if isinstance(x, Dual):
        return Dual(math.tan(x.real), x.dual / math.cos(x.real)**2)
    return Dual(math.tan(x))


def asin(x):
    """arcsin(x) with dy/dx = 1/sqrt(1-x**2)"""
    if isinstance(x, Dual):
        return Dual(math.asin(x.real), x.dual / math.sqrt(1.0 - x.real**2))
    return Dual(math.asin(x))


def acos(x):
    """arccos(x) with dy/dx = -1/sqrt(1-x**2)"""
    if isinstance(x, Dual):
        return Dual(math.acos(x.real), -x.dual / math.sqrt(1.0 - x.real**2))
    return Dual(math.acos(x))


def atan(x):
    """arctan(x) with dy/dx = 1/(x**2+1)"""
    if isinstance(x, Dual):
        return Dual(math.atan(x.real), x.dual / (x.real**2 + 1.0))
    return Dual(math.atan(x))


# Hyperbolic trigonometric functions


acosh = None  # acosh(x) with dydx = 1/(sqrt(x-1)*sqrt(x+1))
asinh = None  # asinh(x) with dydx = 1/sqrt(x**2+1)
atanh = None  # atanh(x) with dydx = 1/(1-x**2)
cosh = None  # cosh(x) with dydx = sinh(x)
sinh = None  # sinh(x) with dydx = cosh(x)
tanh = None  # tanh(x) with dydx = (sech(x))**2


# Classification and other floating point functions


isfinite = None
isinf = None
isnan = None
isclose = None
