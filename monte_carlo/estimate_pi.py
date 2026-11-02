
import random
import math
import matplotlib.pyplot as plt


def estimate_pi(num_samples):
    inside = 0

    for _ in range(num_samples):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            inside += 1

    pi_estimate = 4 * inside / num_samples
    return pi_estimate


def run_experiment():
    sample_sizes = [
        100, 500, 1000, 5000, 10000,
        50000, 100000, 500000
    ]

    estimates = []

    print("Monte Carlo Estimation of Pi")
    print("-" * 45)

    for n in sample_sizes:
        estimate = estimate_pi(n)
        error = abs(estimate - math.pi)

        estimates.append(estimate)

        print(
            f"Samples = {n:7d} | "
            f"Estimate = {estimate:.6f} | "
            f"Error = {error:.6f}"
        )

    # Plot convergence
    plt.figure(figsize=(10, 6))

    plt.plot(sample_sizes, estimates, marker="o",
             label="Monte Carlo Estimate")

    plt.axhline(math.pi, linestyle="--",
                label="Actual Pi")

    plt.xscale("log")
    plt.xlabel("Number of Samples (log scale)")
    plt.ylabel("Estimated Pi")
    plt.title("Monte Carlo Convergence to Pi")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/graphs/monte_carlo_pi.png")
    plt.show()


if __name__ == "__main__":
    run_experiment()
