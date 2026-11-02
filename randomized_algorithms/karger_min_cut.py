
import random
import copy
import matplotlib.pyplot as plt


def karger_min_cut(graph):
    """
    Karger's randomized min-cut algorithm.

    graph: dictionary representing an undirected multigraph.
    Returns the cut size.
    """
    graph = copy.deepcopy(graph)

    while len(graph) > 2:
        u = random.choice(list(graph.keys()))
        v = random.choice(graph[u])

        # Contract vertex v into vertex u
        graph[u].extend(graph[v])

        # Remove self-loops
        graph[u] = [x for x in graph[u] if x != u]

        # Replace references to v with u
        for vertex in graph[v]:
            graph[vertex] = [
                u if x == v else x
                for x in graph[vertex]
            ]

        del graph[v]

    # Remaining edges represent the cut
    return len(list(graph.values())[0])


def run_experiment():
    # A simple undirected graph
    graph = {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2]
    }

    trials = 1000
    min_cut = 3  # Known minimum cut of K4

    successes = 0
    cut_results = []

    for _ in range(trials):
        result = karger_min_cut(graph)
        cut_results.append(result)

        if result == min_cut:
            successes += 1

    probability = successes / trials

    # For a graph with n vertices, success probability
    # in one run is at least 2 / (n * (n - 1)).
    n = len(graph)
    theoretical = 2 / (n * (n - 1))

    print("Karger's Min-Cut Experiment")
    print("-" * 40)
    print("Vertices:", n)
    print("Trials:", trials)
    print("Known minimum cut:", min_cut)
    print("Successful trials:", successes)
    print(f"Observed success probability: {probability:.4f}")
    print(f"Theoretical lower bound: {theoretical:.4f}")

    plt.figure(figsize=(8, 5))

    plt.hist(cut_results, bins=range(
        min(cut_results), max(cut_results) + 2
    ), align="left", rwidth=0.8)

    plt.xlabel("Cut Size")
    plt.ylabel("Frequency")
    plt.title("Karger's Algorithm: Distribution of Cut Sizes")
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig("results/graphs/karger_min_cut.png")
    plt.show()


if __name__ == "__main__":
    run_experiment()
