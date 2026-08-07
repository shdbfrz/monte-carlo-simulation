"""
Entry point: runs all three Monte Carlo simulations, prints numerical
results, and saves visualization plots.

Usage:
    python src/main.py
"""

import math
import os

from monte_carlo import (
    estimate_pi,
    monte_carlo_integrate,
    simulate_scattering_angles,
    convergence_series,
)
from visualize import (
    plot_pi_estimation,
    plot_convergence,
    plot_scattering_histogram,
    plot_integration,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pi_estimation():
    print("=== 1. Estimating pi via Monte Carlo ===")
    for n in (1_000, 100_000, 1_000_000):
        result = estimate_pi(n, seed=42)
        print(f"  N={n:>9,} -> pi ~= {result.estimate:.6f} "
              f"(error: {result.abs_error:.6f}, std_error: {result.std_error:.6f})")

    plot_pi_estimation(5_000, seed=42,
                        save_path=os.path.join(OUTPUT_DIR, "pi_estimation.png"))

    sizes, estimates = convergence_series(n_max=200_000, step=2_000)
    plot_convergence(sizes, estimates,
                      save_path=os.path.join(OUTPUT_DIR, "pi_convergence.png"))
    print()


def run_integration():
    print("=== 2. Monte Carlo integration of f(x) = sin(x) on [0, pi] ===")
    f = math.sin
    true_value = 2.0  # integral of sin(x) from 0 to pi is exactly 2
    for n in (1_000, 100_000):
        result = monte_carlo_integrate(f, 0, math.pi, n, seed=7)
        result.true_value = true_value
        print(f"  N={n:>9,} -> integral ~= {result.estimate:.6f} "
              f"(true = {true_value:.6f}, error: {result.abs_error:.6f}, "
              f"std_error: {result.std_error:.6f})")

    plot_integration(f, 0, math.pi, 500,
                      save_path=os.path.join(OUTPUT_DIR, "mc_integration.png"),
                      label="sin(x)")
    print()


def run_scattering_simulation():
    print("=== 3. Simulated particle scattering angles (Rutherford-like) ===")
    n_particles = 20_000
    angles = simulate_scattering_angles(n_particles, seed=1)
    import statistics
    mean_deg = math.degrees(statistics.mean(angles))
    median_deg = math.degrees(statistics.median(angles))
    large_angle_frac = sum(1 for a in angles if a > math.radians(90)) / n_particles

    print(f"  Simulated {n_particles:,} particles")
    print(f"  Mean scattering angle   : {mean_deg:.2f} deg")
    print(f"  Median scattering angle : {median_deg:.2f} deg")
    print(f"  Fraction with theta > 90 deg (large-angle/'backscatter'): {large_angle_frac*100:.3f}%")

    plot_scattering_histogram(angles,
                               save_path=os.path.join(OUTPUT_DIR, "scattering_histogram.png"))
    print()


if __name__ == "__main__":
    run_pi_estimation()
    run_integration()
    run_scattering_simulation()
    print("All simulations complete. See outputs/ for generated plots.")
