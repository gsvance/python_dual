"""Unit tests for constructing dual numbers."""

import unittest

from dual import Dual


class TestDualConstructors(unittest.TestCase):

    def test_floats(self):
        Dual(1.0, 1.0)
