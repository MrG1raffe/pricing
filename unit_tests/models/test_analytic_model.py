import unittest
from pricing.models.analytic_model import AnalyticModel


class TestAnalyticModel(unittest.TestCase):
    def test_analytic_model(self):
        self.assertTrue("get_vanilla_option_price_analytic" in AnalyticModel.__dict__)


if __name__ == '__main__':
    unittest.main()
