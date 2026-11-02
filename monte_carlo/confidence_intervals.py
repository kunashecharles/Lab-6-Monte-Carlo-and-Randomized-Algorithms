
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Monte Carlo estimation with confidence interval
# --------------------------------------------------

def monte_carlo_confidence_interval(num_samples):
    samples = np.random.uniform(0, 1, num_samples)

    # Estimate integral of f(x) = x^2
    values = samples ** 2

    estimate = np.mean(values)

    sample_variance = np.var(values, ddof=1)

    standard_error = np.sqrt(
        sample_variance / num_samples
    )

    # 95% confidence interval
    margin = 1.96 * standard_error

    lower = estimate - margin
    upper = estimate + margin

    return estimate, standard_error, lower, upper


# --------------------------------------------------
# 2. Calculate required sample size
# --------------------------------------------------

def required_sample_size(std, accuracy, confidence_z=1.96):
    """
    Estimate sample size required for a specified
    margin of error.

    n = (z * sigma / accuracy)^2
    """
    n = (confidence_z * std / accuracy) ** 2

    return int(np.ceil(n))


# --------------------------------------------------
# 3. Run Experiment
# --------------------------------------------------

def run_experiment():
    np.random.seed(42)

    true_value = 1 / 3

    sample_sizes = [
        1000,
        10000,
        100000,
        500000
    ]

    estimates = []
    lower_bounds = []
    upper_bounds = []

    print("Confidence Intervals and Sample Size")
    print("-" * 55)

    print(f"Analytical value: {true_value:.10f}")
    print()

    # --------------------------------------------------
    # Confidence interval experiments
    # --------------------------------------------------

    for n in sample_sizes:

        estimate, se, lower, upper = (
            monte_carlo_confidence_interval(n)
        )

        estimates.append(estimate)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

        print(f"Samples: {n:,}")
        print(f"Estimate:          {estimate:.10f}")
        print(f"Standard error:   {se:.10e}")
        print(f"95% CI:           [{lower:.10f}, {upper:.10f}]")

        if lower <= true_value <= upper:
            print("Analytical value inside CI: Yes")
        else:
            print("Analytical value inside CI: No")

        print()

    # --------------------------------------------------
    # Required sample size calculations
    # --------------------------------------------------

    # For f(x) = x^2, X is uniformly distributed
    # transformed through f(X)=X^2.
    # Its population standard deviation:
    # sqrt(E[X^4] - E[X^2]^2) = sqrt(1/5 - 1/9)

    population_std = np.sqrt(
        1 / 5 - 1 / 9
    )

    accuracies = [
        0.01,
        0.005,
        0.001
    ]

    print("Required Sample Sizes for 95% Confidence")
    print("-" * 55)

    for accuracy in accuracies:

        n_required = required_sample_size(
            population_std, accuracy
        )

        print(
            f"Accuracy ±{accuracy}: "
            f"{n_required:,} samples"
        )

    # --------------------------------------------------
    # Plot confidence intervals
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    x_positions = np.arange(len(sample_sizes))

    errors_lower = [
        estimates[i] - lower_bounds[i]
        for i in range(len(estimates))
    ]

    errors_upper = [
        upper_bounds[i] - estimates[i]
        for i in range(len(estimates))
    ]

    plt.errorbar(
        x_positions,
        estimates,
        yerr=[errors_lower, errors_upper],
        fmt="o",
        capsize=5,
        label="95% Confidence Interval"
    )

    plt.axhline(
        true_value,
        linestyle="--",
        label="Analytical Value"
    )

    plt.xticks(
        x_positions,
        [f"{n:,}" for n in sample_sizes]
    )

    plt.xlabel("Number of Samples")
    plt.ylabel("Estimated Integral")

    plt.title(
        "Monte Carlo Estimates with 95% Confidence Intervals"
    )

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/confidence_intervals.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    run_experiment()
