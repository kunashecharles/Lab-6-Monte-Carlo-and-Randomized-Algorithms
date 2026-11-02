
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def ordinary_monte_carlo(num_samples, threshold, mean, std):
    losses = np.random.normal(mean, std, num_samples)

    indicators = (losses > threshold).astype(float)

    probability = np.mean(indicators)
    variance = np.var(indicators, ddof=1) / num_samples

    return probability, variance


def importance_sampling(
    num_samples, threshold, mean, std, shifted_mean
):
    """
    Estimate a rare-event probability using Importance Sampling.

    Samples are drawn from a shifted normal distribution.
    Likelihood ratios correct the distribution shift.
    """
    samples = np.random.normal(
        shifted_mean, std, num_samples
    )

    # Likelihood ratio: original density / shifted density
    weights = np.exp(
        ((samples - shifted_mean) ** 2
         - (samples - mean) ** 2)
        / (2 * std ** 2)
    )

    indicators = (samples > threshold).astype(float)

    weighted_values = indicators * weights

    probability = np.mean(weighted_values)

    # Variance of the estimator
    variance = np.var(
        weighted_values, ddof=1
    ) / num_samples

    return probability, variance


def confidence_interval(probability, variance):
    """Calculate a 95% confidence interval."""
    standard_error = np.sqrt(variance)

    lower = probability - 1.96 * standard_error
    upper = probability + 1.96 * standard_error

    # Probability cannot be below 0 or above 1
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return lower, upper


def run_experiment():
    np.random.seed(42)

    num_samples = 100000
    mean = 0
    std = 200000
    threshold = 1000000

    # Shift sampling distribution toward the rare event
    shifted_mean = 1000000

    # Run Ordinary Monte Carlo
    ordinary_prob, ordinary_var = ordinary_monte_carlo(
        num_samples, threshold, mean, std
    )

    # Run Importance Sampling
    importance_prob, importance_var = importance_sampling(
        num_samples, threshold, mean, std, shifted_mean
    )

    # Analytical probability for validation
    true_probability = norm.sf(
        (threshold - mean) / std
    )

    # Confidence interval for Importance Sampling
    ci_lower, ci_upper = confidence_interval(
        importance_prob, importance_var
    )

    print("Rare Portfolio Loss Estimation")
    print("-" * 55)

    print(f"Threshold:             ${threshold:,.0f}")
    print(f"Samples:              {num_samples:,}")
    print()

    print(f"Analytical probability: {true_probability:.8e}")
    print(f"Ordinary Monte Carlo:  {ordinary_prob:.8e}")
    print(f"Importance Sampling:    {importance_prob:.8e}")
    print()

    print(f"Ordinary variance:     {ordinary_var:.8e}")
    print(f"Importance variance:    {importance_var:.8e}")
    print()

    print("Importance Sampling 95% Confidence Interval:")
    print(f"Lower bound:           {ci_lower:.8e}")
    print(f"Upper bound:           {ci_upper:.8e}")
    print()

    # Safe variance reduction calculation
    if ordinary_var > 0:

        reduction = (
            1 - importance_var / ordinary_var
        ) * 100

        print(f"Variance reduction:    {reduction:.2f}%")

    else:

        print("Variance reduction:    Undefined")
        print(
            "Reason: Ordinary Monte Carlo observed "
            "zero rare events."
        )
        print(
            "Importance Sampling successfully "
            "estimated the rare-event probability."
        )

    # Plot comparison
    labels = [
        "Ordinary MC",
        "Importance Sampling",
        "Analytical"
    ]

    values = [
        ordinary_prob,
        importance_prob,
        true_probability
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(labels, values)

    plt.ylabel("Estimated Probability")
    plt.title("Rare Event Probability Estimation")

    plt.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0)
    )

    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/importance_sampling.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    run_experiment()
