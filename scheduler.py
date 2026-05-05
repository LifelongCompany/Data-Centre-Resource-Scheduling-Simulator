"""
Scheduler module for Multi-Agent Digital Twin Simulation.
Contains definitions for BaselineScheduler and MASTwinScheduler.
"""

class BaselineScheduler:
    """
    Implements a greedy 'LeastAllocated' strategy.
    Selects the cluster with the maximum available CPU resources.
    Falls back to the first available cluster in case of a tie.
    """
    def __init__(self, clusters):
        """
        Initialize the BaselineScheduler.

        Args:
            clusters (list): A list of ClusterAgent instances.
        """
        self.clusters = clusters

    def schedule(self, task):
        """
        Schedule a task based on raw available CPU capacity.

        Args:
            task (DigitalTwinTaskAgent): Task to be scheduled.

        Returns:
            ClusterAgent: The assigned cluster, or None if no cluster has sufficient capacity.
        """
        best_cluster = None
        max_cpu_available = -1

        for cluster in self.clusters:
            if cluster.can_accept(task):
                if cluster.cpu_available > max_cpu_available:
                    max_cpu_available = cluster.cpu_available
                    best_cluster = cluster

        return best_cluster


class MASTwinScheduler:
    """
    A predictive scheduler that selects clusters with the lowest future congestion risk,
    leveraging Markov chain state predictions.
    Falls back to Baseline logic for tie-breaking.
    """
    def __init__(self, clusters):
        """
        Initialize the MASTwinScheduler.

        Args:
            clusters (list): A list of ClusterAgent instances.
        """
        self.clusters = clusters

    def schedule(self, task):
        """
        Schedule a task aiming to minimize predicted background interference.

        Args:
            task (DigitalTwinTaskAgent): Task to be scheduled.

        Returns:
            ClusterAgent: The assigned cluster, or None if no cluster has sufficient capacity.
        """
        best_cluster = None
        min_congestion_risk = float('inf')
        max_cpu_available = -1

        for cluster in self.clusters:
            if cluster.can_accept(task):
                # Predict 1 step ahead future state probabilities
                future_probs = cluster.bg_load_generator.predict_future_state(1)

                # Congestion risk: probability of entering State 2 (High) or State 3 (Overload)
                congestion_risk = future_probs[2] + future_probs[3]

                if congestion_risk < min_congestion_risk:
                    min_congestion_risk = congestion_risk
                    max_cpu_available = cluster.cpu_available
                    best_cluster = cluster
                elif congestion_risk == min_congestion_risk:
                    # Tie-breaking: revert to Baseline logic (max CPU available)
                    if cluster.cpu_available > max_cpu_available:
                        max_cpu_available = cluster.cpu_available
                        best_cluster = cluster

        return best_cluster
