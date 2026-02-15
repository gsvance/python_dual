"""Unit tests for converting dual numbers into other numerical types."""

import unittest

from dual import Dual


class TestDualConversions(unittest.TestCase):

    def test_repr(self):
        self.assertEqual(int(Dual(2.0, 0.0)), 2)
