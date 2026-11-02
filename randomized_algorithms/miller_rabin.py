
import random


def miller_rabin(n, k=5):
    """
    Probabilistic primality test.
    k is the number of independent rounds.
    """
    if n < 2:
        return False

    if n in (2, 3):
        return True

    if n % 2 == 0:
        return False

    # Express n - 1 as 2^r * d, where d is odd
    d = n - 1
    r = 0

    while d % 2 == 0:
        d //= 2
        r += 1

    # Perform k independent randomized rounds
    for _ in range(k):
        a = random.randrange(2, n - 1)

        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)

            if x == n - 1:
                break
        else:
            return False

    return True



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Carmichael numbers are composite but can fool
    # some primality tests with particular bases.
    test_number = 561

    rounds = [1, 2, 3, 4, 5, 6, 8, 10]
    trials = 1000
    error_rates = []

    print("Miller-Rabin Error Probability Experiment")
    print("-" * 50)

    for k in rounds:
        errors = 0

        for _ in range(trials):
            result = miller_rabin(test_number, k=k)

            if result:  # Incorrectly declared composite as prime
                errors += 1

        error_rate = errors / trials
        error_rates.append(error_rate)

        theoretical = (1 / 4) ** k

        print(
            f"Rounds = {k:2d} | "
            f"Observed error = {error_rate:.5f} | "
            f"Theoretical upper bound = {theoretical:.8f}"
        )

    plt.figure(figsize=(9, 5))

    plt.plot(rounds, error_rates, marker="o",
             label="Observed Error Probability")

    theoretical = [(1 / 4) ** k for k in rounds]

    plt.plot(rounds, theoretical, linestyle="--",
             label="Theoretical Upper Bound")

    plt.yscale("log")
    plt.xlabel("Number of Rounds (k)")
    plt.ylabel("Error Probability (log scale)")
    plt.title("Miller-Rabin: Error Probability vs. Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/graphs/miller_rabin_error.png")
    plt.show()
