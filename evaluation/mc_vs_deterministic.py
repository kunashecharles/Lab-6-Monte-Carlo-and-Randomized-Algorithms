
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.integrate import cubature


# --------------------------------------------------
# 1. Define the 5-dimensional function
# --------------------------------------------------

def function_5d(x):
    """
    x has shape (number_of_points, 5).

    f(x) = x1^2 + x2^2 + x3^2 + x4^2 + x5^2
    """
    return np.sum(x ** 2, axis=1)


# --------------------------------------------------
# 2. Monte Carlo Integration
# --------------------------------------------------

def monte_carlo_integration(num_samples):
    start_time = time.perf_counter()

    samples = np.random.uniform(
        0, 1, size=(num_samples, 5)
    )

    values = function_5d(samples)

    estimate = np.mean(values)

    elapsed_time = time.perf_counter() - start_time

    return estimate, elapsed_time


# --------------------------------------------------
# 3. Deterministic Numerical Integration
# --------------------------------------------------

def deterministic_integration():
    start_time = time.perf_counter()

    # Correct vectorized function for SciPy cubature
    def integrand(x):
        return np.sum(x ** 2, axis=1)

    result = cubature(
        f=integrand,
        a=np.zeros(5),
        b=np.ones(5),
        rule="genz-malik",
        rtol=1e-6,
        atol=1e-10,
        max_subdivisions=10000
    )

    elapsed_time = time.perf_counter() - start_time

    return float(result.estimate), elapsed_time


# --------------------------------------------------
# 4. Run Experiment
# --------------------------------------------------

def run_experiment():
    np.random.seed(42)

    # Exact analytical integral
    true_value = 5 / 3

    sample_sizes = [
        1000,
        10000,
        100000,
        500000
    ]

    mc_estimates = []
    mc_times = []
    mc_errors = []

    print("Monte Carlo vs Deterministic Integration")
    print("-" * 60)

    print("Dimensions:          5")
    print(f"Analytical value:   {true_value:.10f}")
    print()

    # --------------------------------------------------
    # Monte Carlo experiments
    # --------------------------------------------------

    for n in sample_sizes:

        estimate, elapsed_time = (
            monte_carlo_integration(n)
        )

        error = abs(estimate - true_value)

        mc_estimates.append(estimate)
        mc_times.append(elapsed_time)
        mc_errors.append(error)

        print(f"Monte Carlo (N={n:,})")
        print(f"Estimate:  {estimate:.10f}")
        print(f"Error:    {error:.10e}")
        print(f"Time:     {elapsed_time:.6f} seconds")
        print()

    # --------------------------------------------------
    # Deterministic integration
    # --------------------------------------------------

    det_estimate, det_time = (
        deterministic_integration()
    )

    det_error = abs(det_estimate - true_value)

    print("Deterministic Cubature")
    print(f"Estimate:  {det_estimate:.10f}")
    print(f"Error:    {det_error:.10e}")
    print(f"Time:     {det_time:.6f} seconds")
    print()

    # --------------------------------------------------
    # 5. Plot estimation error comparison
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        sample_sizes,
        mc_errors,
        marker="o",
        label="Monte Carlo"
    )

    plt.axhline(
        det_error,
        linestyle="--",
        label="Deterministic Cubature"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Number of Monte Carlo Samples")
    plt.ylabel("Absolute Error")

    plt.title(
        "Monte Carlo vs Deterministic Integration"
    )

    plt.legend()

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        "results/graphs/mc_vs_deterministic.png",
        dpi=300
    )

    plt.show()

    # --------------------------------------------------
    # 6. Plot execution time comparison
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        sample_sizes,
        mc_times,
        marker="o",
        label="Monte Carlo"
    )

    plt.axhline(
        det_time,
        linestyle="--",
        label="Deterministic Cubature"
    )

    plt.xscale("log")

    plt.xlabel("Number of Monte Carlo Samples")
    plt.ylabel("Execution Time (seconds)")

    plt.title(
        "Execution Time Comparison"
    )

    plt.legend()

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        "results/graphs/mc_vs_deterministic_time.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    run_experiment()
