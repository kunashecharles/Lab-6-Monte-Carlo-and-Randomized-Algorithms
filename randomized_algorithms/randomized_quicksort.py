
import random
import time
import matplotlib.pyplot as plt


def randomized_quick_sort(arr):
    """Randomized Quick Sort with comparison counting."""
    comparisons = 0

    def sort(data):
        nonlocal comparisons

        if len(data) <= 1:
            return data

        pivot = random.choice(data)
        smaller = []
        equal = []
        larger = []

        for item in data:
            comparisons += 1

            if item < pivot:
                smaller.append(item)
            elif item > pivot:
                larger.append(item)
            else:
                equal.append(item)

        return sort(smaller) + equal + sort(larger)

    result = sort(arr)
    return result, comparisons


def benchmark():
    sizes = [100, 500, 1000, 2000, 5000, 10000]
    operation_counts = []
    execution_times = []

    for n in sizes:
        data = [random.randint(1, 100000) for _ in range(n)]

        start = time.perf_counter()
        sorted_data, comparisons = randomized_quick_sort(data)
        end = time.perf_counter()

        operation_counts.append(comparisons)
        execution_times.append(end - start)

        print(f"n = {n:5d} | Comparisons = {comparisons:8d} "
              f"| Time = {end - start:.6f}s "
              f"| Correct = {sorted_data == sorted(data)}")

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, operation_counts, marker="o",
             label="Experimental Comparisons")

    theoretical = [n * (n.bit_length() - 1) for n in sizes]
    plt.plot(sizes, theoretical, linestyle="--",
             label="n log₂ n (Reference)")

    plt.xlabel("Input Size (n)")
    plt.ylabel("Number of Comparisons")
    plt.title("Randomized Quick Sort: Empirical Analysis")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/graphs/randomized_quicksort.png")
    plt.show()


if __name__ == "__main__":
    benchmark()
