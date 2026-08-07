"""Visualization helpers for the Monte Carlo simulations."""

import math
import random

import matplotlib.pyplot as plt
import numpy as np


def plot_pi_estimation(n_samples: int, seed: int, save_path: str):
    """Scatter plot of sample points inside/outside the unit circle."""
    random.seed(seed)
    xs_in, ys_in, xs_out, ys_out = [], [], [], []
    for _ in range(n_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if x * x + y * y <= 1.0:
            xs_in.append(x); ys_in.append(y)
        else:
            xs_out.append(x); ys_out.append(y)

    pi_est = 4 * len(xs_in) / n_samples

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs_in, ys_in, s=4, color="#2f6fb3", label="Inside circle")
    ax.scatter(xs_out, ys_out, s=4, color="#e8944a", label="Outside circle")
    circle = plt.Circle((0, 0), 1, fill=False, color="black", linewidth=1.5)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_title(f"Monte Carlo Estimation of \u03c0\nN={n_samples:,} samples \u2192 \u03c0 \u2248 {pi_est:.5f} (true \u03c0 = {math.pi:.5f})")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_convergence(sizes, estimates, save_path: str):
    """Plot how the pi estimate converges to the true value as N grows,
    alongside the theoretical 1/sqrt(N) error envelope."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sizes, estimates, color="#2f6fb3", linewidth=1.2, label="Monte Carlo estimate")
    ax.axhline(math.pi, color="#c0392b", linestyle="--", linewidth=1.2, label="True \u03c0")

    sizes_arr = np.array(sizes, dtype=float)
    envelope = 4.0 * np.sqrt(0.25 * 0.75 / sizes_arr)  # approx std error near p=pi/4
    ax.fill_between(sizes, math.pi - envelope, math.pi + envelope,
                     color="#c0392b", alpha=0.12, label="~1 std-error envelope")

    ax.set_xlabel("Number of samples (N)")
    ax.set_ylabel("Estimated value of \u03c0")
    ax.set_title("Monte Carlo Convergence: Estimate vs. Sample Size")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_scattering_histogram(angles, save_path: str):
    """Histogram of simulated scattering angles (in degrees)."""
    degrees = [math.degrees(a) for a in angles]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(degrees, bins=60, color="#2f6fb3", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Scattering angle \u03b8 (degrees)")
    ax.set_ylabel("Number of particles (log scale)")
    ax.set_title(f"Simulated Particle Scattering Angle Distribution (N={len(angles):,})\n"
                 f"Sampled from dN/d\u03b8 \u221d 1/sin\u2074(\u03b8/2) \u2014 Rutherford-like small-angle dominance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_integration(f, a, b, n_points, save_path: str, label="f(x)"):
    """Visualize the function being integrated with a sample of MC points
    used to estimate the integral."""
    xs = np.linspace(a, b, 400)
    ys = [f(x) for x in xs]

    random_xs = [random.uniform(a, b) for _ in range(min(400, n_points))]
    random_ys = [f(x) for x in random_xs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, color="#2f6fb3", linewidth=2, label=label)
    ax.fill_between(xs, 0, ys, color="#2f6fb3", alpha=0.15)
    ax.scatter(random_xs, random_ys, s=8, color="#e8944a", alpha=0.6, label="MC sample points")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Monte Carlo Integration of {label} on [{a}, {b}]")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
