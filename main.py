
import subprocess
import sys


# --------------------------------------------------
# Lab 6: Monte Carlo Simulation and Randomized Algorithms
# Main Execution Program
# --------------------------------------------------

def run_script(script_path):
    print("\n" + "=" * 65)
    print(f"Running: {script_path}")
    print("=" * 65)

    result = subprocess.run(
        [sys.executable, script_path]
    )

    if result.returncode != 0:
        print(f"\nERROR: {script_path} failed.")
        return False

    return True


def main():

    print("=" * 65)
    print("LAB 6: MONTE CARLO SIMULATION AND RANDOMIZED ALGORITHMS")
    print("=" * 65)

    scripts = [

        # Randomized Algorithms
        "randomized_algorithms/randomized_quicksort.py",
        "randomized_algorithms/miller_rabin.py",
        "randomized_algorithms/karger_min_cut.py",

        # Monte Carlo Simulation
        "monte_carlo/estimate_pi.py",
        "monte_carlo/mm1_queue.py",
        "monte_carlo/importance_sampling.py",

        # Variance Reduction
        "variance_reduction/variance_reduction.py",

        # Evaluation
        "evaluation/mc_vs_deterministic.py",

        # Statistical Analysis
        "monte_carlo/confidence_intervals.py"
    ]

    for script in scripts:

        success = run_script(script)

        if not success:
            print("\nExecution stopped due to an error.")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("ALL LAB 6 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
