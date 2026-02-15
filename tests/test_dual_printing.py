"""Unit tests for turning dual numbers into strings."""

import unittest

from dual import Dual, set_epsilon_variant


class TestDualPrinting(unittest.TestCase):

    def test_repr(self):
        self.assertEqual(repr(Dual(5.0, -1.0)), 'Dual(5.0, -1.0)')
