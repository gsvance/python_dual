"""Unit tests for doing arithmetic with dual numbers."""

import unittest

from dual import Dual


class TestDualArithmetic(unittest.TestCase):

    def test_add(self):
        self.assertEqual(Dual(1.0, 2.0) + Dual(3.0, 4.0), Dual(4.0, 6.0))
