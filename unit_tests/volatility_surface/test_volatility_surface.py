import unittest
import numpy as np
from scipy.stats import norm

from pricing.volatility_surface.volatility_surface import black_iv


class TestBlackIV(unittest.TestCase):
    def test_call(self):
        """
        Calculate the price of call option with the Black-76 formula and check the IV.
        """
        r = 0.05
        sigma = 0.2
        F0 = 96
        K = np.array([
            [95, 96, 97],
            [94, 96, 98],
            [91, 96, 101],
            [90, 96, 102]
        ])
        T = np.array([0.15, 0.5, 0.75, 1])
        T = np.reshape(T, (-1, 1))

        d1 = (np.log(F0 / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_prices = np.exp(-r * T) * (F0 * norm.cdf(d1) - K * norm.cdf(d2))

        self.assertTrue(np.allclose(
            black_iv(option_price=call_prices, T=T, K=K, F=F0, r=r, flag='c'),
            np.ones_like(call_prices) * sigma
        ))

    def test_put(self):
        """
        Calculate the price of put option with the Black-76 formula and check the IV.
        """
        r = 0.1
        sigma = 0.4
        F0 = 96.5
        K = np.array([
            [95, 96.5, 97],
            [95, 96, 97],
            [92, 96, 103],
            [91, 96, 102]
        ])
        T = np.array([0.05, 0.4, 0.3, 0.8])
        T = np.reshape(T, (-1, 1))

        d1 = (np.log(F0 / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_prices = np.exp(-r * T) * (-F0 * norm.cdf(-d1) + K * norm.cdf(-d2))

        self.assertTrue(np.allclose(
            black_iv(option_price=put_prices, T=T, K=K, F=F0, r=r, flag='p'),
            np.ones_like(put_prices) * sigma
        ))


if __name__ == '__main__':
    unittest.main()
