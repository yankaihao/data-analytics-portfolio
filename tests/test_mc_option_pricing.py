import unittest
from unittest.mock import patch


from mc_option_pricing import BlackScholesMarket, estimate_greeks_finite_diff, simulate_gbm_trajectories


class MonteCarloTests(unittest.TestCase):
    def test_seed_reproducible_and_estimators_reasonable(self):
        market = BlackScholesMarket()
        result = simulate_gbm_trajectories(market, 100_000, 2022)
        self.assertEqual(result, simulate_gbm_trajectories(market, 100_000, 2022))
        for name, value in result.items():
            with self.subTest(name=name):
                self.assertLess(abs(value['price'] - market.exact_price()), 4 * value['stderr'])
        self.assertLess(result['joint_av_cv']['stderr'], result['plain']['stderr'])
        # Independent plain draws no longer produce the same mean as paired draws.
        self.assertNotEqual(result['plain']['price'], result['antithetic']['price'])

    def test_input_validation(self):
        for n in (0, 3, 101):
            with self.assertRaises(ValueError):
                simulate_gbm_trajectories(BlackScholesMarket(), n)
        with self.assertRaises(ValueError):
            simulate_gbm_trajectories(BlackScholesMarket(steps=1), 100)

    def test_greeks_reuse_seed_and_match_analytic(self):
        import mc_option_pricing as mc
        with patch.object(mc, 'simulate_gbm_trajectories', wraps=simulate_gbm_trajectories) as simulate:
            greeks = estimate_greeks_finite_diff(BlackScholesMarket(), 100_000, 2022)
        self.assertEqual([call.args[2] for call in simulate.call_args_list], [2022] * 4)
        self.assertAlmostEqual(greeks['delta_mc'], greeks['delta_exact'], delta=0.01)
        self.assertAlmostEqual(greeks['vega_mc'], greeks['vega_exact'], delta=1.0)


if __name__ == '__main__':
    unittest.main()
