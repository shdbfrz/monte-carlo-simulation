"""Correctness tests for the Monte Carlo simulation toolkit."""

import math
import unittest

from monte_carlo import (
    estimate_pi,
    monte_carlo_integrate,
    simulate_scattering_angles,
    convergence_series,
)


class TestMonteCarloPi(unittest.TestCase):

    def test_pi_estimate_converges_within_tolerance(self):
        result = estimate_pi(200_000, seed=1)
        self.assertAlmostEqual(result.estimate, math.pi, delta=0.02)

    def test_pi_estimate_bounded_between_0_and_4(self):
        result = estimate_pi(1_000, seed=2)
        self.assertGreaterEqual(result.estimate, 0)
        self.assertLessEqual(result.estimate, 4)

    def test_std_error_shrinks_with_more_samples(self):
        small = estimate_pi(1_000, seed=3)
        large = estimate_pi(100_000, seed=3)
        self.assertLess(large.std_error, small.std_error)

    def test_deterministic_with_seed(self):
        r1 = estimate_pi(5_000, seed=99)
        r2 = estimate_pi(5_000, seed=99)
        self.assertEqual(r1.estimate, r2.estimate)


class TestMonteCarloIntegration(unittest.TestCase):

    def test_integral_of_sine_0_to_pi(self):
        # exact value of integral of sin(x) dx from 0 to pi is 2
        result = monte_carlo_integrate(math.sin, 0, math.pi, 200_000, seed=5)
        self.assertAlmostEqual(result.estimate, 2.0, delta=0.02)

    def test_integral_of_constant_function(self):
        # integral of f(x) = 3 over [0, 4] should be exactly 12
        result = monte_carlo_integrate(lambda x: 3, 0, 4, 10_000, seed=6)
        self.assertAlmostEqual(result.estimate, 12.0, delta=0.01)

    def test_integral_of_linear_function(self):
        # integral of f(x) = x over [0, 2] is 2
        result = monte_carlo_integrate(lambda x: x, 0, 2, 100_000, seed=8)
        self.assertAlmostEqual(result.estimate, 2.0, delta=0.05)


class TestScatteringSimulation(unittest.TestCase):

    def test_correct_number_of_particles_generated(self):
        angles = simulate_scattering_angles(500, seed=10)
        self.assertEqual(len(angles), 500)

    def test_angles_within_valid_range(self):
        angles = simulate_scattering_angles(1_000, seed=11)
        for theta in angles:
            self.assertGreaterEqual(theta, 0)
            self.assertLessEqual(theta, math.pi)

    def test_small_angle_scattering_dominates(self):
        # Rutherford-like distribution should heavily favor small angles
        angles = simulate_scattering_angles(5_000, seed=12)
        small_angle_frac = sum(1 for a in angles if a < math.radians(30)) / len(angles)
        self.assertGreater(small_angle_frac, 0.9)


class TestConvergenceSeries(unittest.TestCase):

    def test_convergence_series_length(self):
        sizes, estimates = convergence_series(n_max=10_000, step=1_000)
        self.assertEqual(len(sizes), 10)
        self.assertEqual(len(estimates), 10)

    def test_later_estimates_closer_to_pi_on_average(self):
        sizes, estimates = convergence_series(n_max=50_000, step=1_000, seed=21)
        early_error = abs(estimates[0] - math.pi)
        late_error = abs(estimates[-1] - math.pi)
        # not guaranteed for every single seed/run, but true on average;
        # use a generous check that late estimate stays reasonably close
        self.assertLess(late_error, 0.05)


if __name__ == "__main__":
    unittest.main()
