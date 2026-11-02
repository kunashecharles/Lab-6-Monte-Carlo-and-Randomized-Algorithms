
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Standard Monte Carlo
# --------------------------------------------------

def standard_monte_carlo(num_samples):
    samples = np.random.uniform(0, 1, num_samples)

    # Estimate the integral of f(x) = x^2
    values = samples ** 2

    estimate = np.mean(values)
    variance = np.var(values, ddof=1) / num_samples

    return estimate, variance


# --------------------------------------------------
# 2. Antithetic Variates
# --------------------------------------------------

def antithetic_variates(num_samples):
    half_samples = num_samples // 2

    u = np.random.uniform(0, 1, half_samples)

    # Antithetic pairs: U and 1-U
    paired_values = (
        u ** 2 + (1 - u) ** 2
    ) / 2

    estimate = np.mean(paired_values)

    variance = (
        np.var(paired_values, ddof=1) / half_samples
    )

    return estimate, variance


# --------------------------------------------------
# 3. Control Variates
# --------------------------------------------------

def control_variates(num_samples):
    samples = np.random.uniform(0, 1, num_samples)

    # Main function: f(x) = x^2
    x_squared = samples ** 2

    # Control variable: g(x) = x
    # Known expected value E[X] = 1/2
    control = samples

    expected_control = 0.5

    # Estimate optimal coefficient
    covariance = np.cov(
        x_squared, control, ddof=1
    )[0, 1]

    control_variance = np.var(
        control, ddof=1
    )

    coefficient = covariance / control_variance

    # Adjust using the known expected value
    adjusted_values = (
        x_squared
        - coefficient * (control - expected_control)
    )

    estimate = np.mean(adjusted_values)

    variance = (
        np.var(adjusted_values, ddof=1) / num_samples
    )

    return estimate, variance


# --------------------------------------------------
# 4. Run Experiment
# --------------------------------------------------

def run_experiment():
    np.random.seed(42)

    num_samples = 100000

    # Analytical answer: integral from 0 to 1 of x^2
    true_value = 1 / 3

    # Run all methods
    standard_estimate, standard_variance = (
        standard_monte_carlo(num_samples)
    )

    antithetic_estimate, antithetic_variance = (
        antithetic_variates(num_samples)
    )

    control_estimate, control_variance = (
        control_variates(num_samples)
    )

    # Calculate percentage variance reduction
    antithetic_reduction = (
        1 - antithetic_variance / standard_variance
    ) * 100

    control_reduction = (
        1 - control_variance / standard_variance
    ) * 100

    # Print results
    print("Variance Reduction Techniques")
    print("-" * 55)

    print(f"Number of samples: {num_samples:,}")
    print(f"Analytical value:  {true_value:.10f}")
    print()

    print("Standard Monte Carlo")
    print(f"Estimate:  {standard_estimate:.10f}")
    print(f"Variance:  {standard_variance:.10e}")
    print()

    print("Antithetic Variates")
    print(f"Estimate:  {antithetic_estimate:.10f}")
    print(f"Variance:  {antithetic_variance:.10e}")
    print(f"Variance reduction: {antithetic_reduction:.2f}%")
    print()

    print("Control Variates")
    print(f"Estimate:  {control_estimate:.10f}")
    print(f"Variance:  {control_variance:.10e}")
    print(f"Variance reduction: {control_reduction:.2f}%")

    # Plot variance comparison
    methods = [
        "Standard MC",
        "Antithetic",
        "Control Variates"
    ]

    variances = [
        standard_variance,
        antithetic_variance,
        control_variance
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(methods, variances)

    plt.ylabel("Estimator Variance")
    plt.title("Variance Reduction Comparison")

    plt.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )

    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/variance_reduction.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    run_experiment()
