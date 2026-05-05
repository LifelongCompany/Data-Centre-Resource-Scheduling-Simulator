"""
Scheduler module for Multi-Agent Digital Twin Simulation.
Contains definitions for BaselineScheduler and MASTwinScheduler.
"""

class BaselineScheduler:
    """
    BaselineScheduler implements a greedy 'LeastAllocated' approach.
    It selects the cluster with the maximum available CPU resources.
    If tied, it selects the first available cluster in the list.
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
        Schedule a task to a cluster based on available CPU.

        Args:
            task (DigitalTwinTaskAgent): The task to be scheduled.

        Returns:
            ClusterAgent: The selected cluster, or None if no cluster can accept the task.
        """
        best_cluster = None
        max_cpu_available = -1

        for cluster in self.clusters:
            if cluster.can_accept(task):
                if cluster.cpu_available > max_cpu_available:
                    max_cpu_available = cluster.cpu_available
                    best_cluster = cluster
                # Implicitly handles tie-breaking by not updating if equal

        return best_cluster


class MASTwinScheduler:
    """
    MASTwinScheduler selects the cluster with the lowest congestion risk based on Markov predictions.
    If tied, it falls back to the Baseline logic (max available CPU, then first available).
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
        Schedule a task to a cluster based on lowest congestion risk.

        Args:
            task (DigitalTwinTaskAgent): The task to be scheduled.

        Returns:
            ClusterAgent: The selected cluster, or None if no cluster can accept the task.
        """
        best_cluster = None
        min_congestion_risk = float('inf')
        max_cpu_available = -1

        for cluster in self.clusters:
            if cluster.can_accept(task):
                # Predict future state probabilities for 1 step ahead
                future_probs = cluster.bg_load_generator.predict_future_state(1)

                # Congestion risk is the sum of probabilities of entering State 2 (High) or State 3 (Overload)
                # Assuming state index 2 is High and 3 is Overload as per InterferenceModel and MarkovEnv
                congestion_risk = future_probs[2] + future_probs[3]

                if congestion_risk < min_congestion_risk:
                    min_congestion_risk = congestion_risk
                    max_cpu_available = cluster.cpu_available
                    best_cluster = cluster
                elif congestion_risk == min_congestion_risk:
                    # Fallback to Baseline logic: max available CPU
                    if cluster.cpu_available > max_cpu_available:
                        max_cpu_available = cluster.cpu_available
                        best_cluster = cluster
                        # Tied CPU keeps the first one implicitly

        return best_cluster
