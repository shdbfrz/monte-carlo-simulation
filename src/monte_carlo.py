"""
Monte Carlo Simulation Toolkit
-------------------------------
Three classic Monte Carlo methods used across scientific computing and
particle physics research (the kind of numerical technique used
extensively at CERN for detector simulation, cross-section estimation,
and uncertainty propagation):

1. Estimating pi by random sampling (geometric Monte Carlo)
2. Monte Carlo numerical integration of an arbitrary function
3. A simplified particle-scattering simulation, sampling scattering
   angles from a probability distribution (inspired by Rutherford
   scattering) to build an angular distribution histogram

All methods rely on the Law of Large Numbers: as the number of random
samples N grows, the Monte Carlo estimate converges to the true value,
with statistical error shrinking proportionally to 1/sqrt(N).
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class MCResult:
    estimate: float
    true_value: float = None
    n_samples: int = 0
    std_error: float = None

    @property
    def abs_error(self):
        if self.true_value is None:
            return None
        return abs(self.estimate - self.true_value)


def estimate_pi(n_samples: int, seed: int = None) -> MCResult:
    """Estimate pi by sampling random points in a unit square and
    checking what fraction fall inside the inscribed unit circle.

    P(inside circle) = area(circle) / area(square) = (pi * r^2) / (2r)^2 = pi / 4
    => pi ~= 4 * (points_inside / total_points)
    """
    if seed is not None:
        random.seed(seed)

    inside = 0
    for _ in range(n_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x * x + y * y <= 1.0:
            inside += 1

    estimate = 4.0 * inside / n_samples
    p = inside / n_samples
    # Standard error of a proportion, propagated to the pi estimate (x4)
    std_error = 4.0 * math.sqrt(p * (1 - p) / n_samples)

    return MCResult(estimate=estimate, true_value=math.pi,
                     n_samples=n_samples, std_error=std_error)


def monte_carlo_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int,
    seed: int = None,
) -> MCResult:
    """Estimate the definite integral of f over [a, b] using simple
    (uniform-sampling) Monte Carlo integration:

        integral ~= (b - a) * mean(f(x_i)) for x_i ~ Uniform(a, b)

    Also returns the standard error of the estimate based on the
    sample variance of f(x), which shrinks as 1/sqrt(N).
    """
    if seed is not None:
        random.seed(seed)

    samples = [f(random.uniform(a, b)) for _ in range(n_samples)]
    mean = sum(samples) / n_samples
    variance = sum((s - mean) ** 2 for s in samples) / (n_samples - 1)

    estimate = (b - a) * mean
    std_error = (b - a) * math.sqrt(variance / n_samples)

    return MCResult(estimate=estimate, n_samples=n_samples, std_error=std_error)


def simulate_scattering_angles(
    n_particles: int,
    seed: int = None,
) -> List[float]:
    """Simulate scattering angles for n_particles using inverse-transform
    sampling from a simplified Rutherford-like differential cross-section:

        dN/d(theta) proportional to 1 / sin^4(theta / 2)

    This heavily favors small-angle scattering with a long tail of rare
    large-angle deflections -- the same qualitative shape (small-angle
    dominance, rare large-angle events) that motivated Rutherford's
    discovery of the atomic nucleus. We sample from a bounded, normalized
    approximation of this distribution over theta in (theta_min, pi].

    Returns a list of scattering angles in radians.
    """
    if seed is not None:
        random.seed(seed)

    theta_min = 0.05  # avoid the non-integrable singularity at theta = 0
    angles = []
    # Rejection sampling against the (unnormalized) Rutherford shape
    weight_max = 1.0 / math.sin(theta_min / 2) ** 4

    while len(angles) < n_particles:
        theta = random.uniform(theta_min, math.pi)
        weight = 1.0 / math.sin(theta / 2) ** 4
        u = random.uniform(0, weight_max)
        if u <= weight:
            angles.append(theta)

    return angles


def convergence_series(
    n_max: int,
    step: int,
    seed: int = 7,
) -> Tuple[List[int], List[float]]:
    """Run estimate_pi repeatedly at increasing sample sizes to show
    Monte Carlo convergence behaviour (error shrinking as 1/sqrt(N))."""
    random.seed(seed)
    sizes, estimates = [], []
    for n in range(step, n_max + 1, step):
        result = estimate_pi(n)
        sizes.append(n)
        estimates.append(result.estimate)
    return sizes, estimates
