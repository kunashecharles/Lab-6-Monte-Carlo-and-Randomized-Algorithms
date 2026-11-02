
import random
import numpy as np
import matplotlib.pyplot as plt


def simulate_mm1(arrival_rate, service_rate, num_customers):
    """Simulate an M/M/1 queue using FCFS scheduling."""

    arrival_times = []
    service_times = []
    waiting_times = []
    system_times = []

    current_arrival = 0.0
    server_available = 0.0

    for _ in range(num_customers):
        # Interarrival time is exponentially distributed
        interarrival = random.expovariate(arrival_rate)
        current_arrival += interarrival

        # Service time is exponentially distributed
        service_time = random.expovariate(service_rate)

        service_start = max(current_arrival, server_available)
        service_end = service_start + service_time

        waiting_time = service_start - current_arrival
        system_time = service_end - current_arrival

        server_available = service_end

        arrival_times.append(current_arrival)
        service_times.append(service_time)
        waiting_times.append(waiting_time)
        system_times.append(system_time)

    return np.mean(waiting_times), np.mean(system_times)


def run_experiment():
    arrival_rate = 4
    service_rate = 6
    num_customers = 100000

    simulated_wait, simulated_system = simulate_mm1(
        arrival_rate, service_rate, num_customers
    )

    # Analytical M/M/1 results
    rho = arrival_rate / service_rate

    theoretical_wait = rho / (service_rate - arrival_rate)
    theoretical_system = 1 / (service_rate - arrival_rate)

    print("M/M/1 Queue Simulation")
    print("-" * 45)
    print("Arrival rate (lambda):", arrival_rate)
    print("Service rate (mu):", service_rate)
    print("Utilization (rho):", round(rho, 4))
    print()
    print("Average Waiting Time:")
    print(f"Simulated:   {simulated_wait:.4f}")
    print(f"Theoretical:  {theoretical_wait:.4f}")
    print()
    print("Average Time in System:")
    print(f"Simulated:   {simulated_system:.4f}")
    print(f"Theoretical:  {theoretical_system:.4f}")

    labels = ["Waiting Time", "Time in System"]
    simulated = [simulated_wait, simulated_system]
    theoretical = [theoretical_wait, theoretical_system]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(x - width / 2, simulated, width,
            label="Simulation")

    plt.bar(x + width / 2, theoretical, width,
            label="Analytical")

    plt.xticks(x, labels)
    plt.ylabel("Average Time")
    plt.title("M/M/1 Queue: Simulation vs Analytical Results")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig("results/graphs/mm1_queue_comparison.png")
    plt.show()


if __name__ == "__main__":
    run_experiment()
