"""Unit tests for comparing dual numbers with each other."""

import unittest

from dual import Dual


class TestDualComparisons(unittest.TestCase):

    def test_eq(self):
        self.assertTrue(Dual(1.0, 2.0) == Dual(1.0, 2.0))
