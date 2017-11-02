import unittest

from data.__main__ import Data


class ClassDataTestCase(unittest.TestCase):
    """Test for data/__main__.py"""

    def test_instantiate(self):
        data = Data()
        self.assertIsInstance(data, Data)

    def test_test(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
