import markov_env

class ClusterAgent:
    """
    Represents a compute cluster resource pool.
    Manages total resources, available resources, and background load.
    """
    def __init__(self, cluster_id, bg_load_generator, cpu_total=10000, gpu_total=10000):
        """
        Initialize the ClusterAgent.

        Args:
            cluster_id (int): Identifier for the cluster.
            bg_load_generator (MarkovClusterGenerator): Instance to predict background load.
            cpu_total (int): Total CPU cores available. Defaults to 10000.
            gpu_total (int): Total GPU cores available. Defaults to 10000.
        """
        self.cluster_id = cluster_id
        self.cpu_total = cpu_total
        self.gpu_total = gpu_total

        self.cpu_available = cpu_total
        self.gpu_available = gpu_total

        self.bg_load_generator = bg_load_generator

    def can_accept(self, task):
        """
        Check if the cluster has enough available resources to accept the task.

        Args:
            task (DigitalTwinTaskAgent): The task to evaluate.

        Returns:
            bool: True if resources are sufficient, False otherwise.
        """
        return (self.cpu_available >= task.cpu_req) and (self.gpu_available >= task.gpu_req)

    def allocate(self, task):
        """
        Allocate resources for the given task.

        Args:
            task (DigitalTwinTaskAgent): The task to allocate resources for.

        Raises:
            ValueError: If the cluster lacks sufficient resources for allocation.
        """
        if self.can_accept(task):
            self.cpu_available -= task.cpu_req
            self.gpu_available -= task.gpu_req
        else:
            raise ValueError(f"Cluster {self.cluster_id} has insufficient resources for task {task.task_id}")

    def release(self, task):
        """
        Release resources previously allocated to a task.

        Args:
            task (DigitalTwinTaskAgent): The completed task whose resources are to be released.
        """
        self.cpu_available += task.cpu_req
        self.gpu_available += task.gpu_req

        # Ensure we do not release more resources than initially configured
        if self.cpu_available > self.cpu_total:
            self.cpu_available = self.cpu_total
        if self.gpu_available > self.gpu_total:
            self.gpu_available = self.gpu_total

    def step_background_load(self):
        """
        Advance the background load state.

        Returns:
            int: The new background load state index.
        """
        return self.bg_load_generator.step()

class DigitalTwinTaskAgent:
    """
    Represents an individual compute task within the digital twin system.
    Tracks resource requirements, timing details, and execution state.
    """
    def __init__(self, task_id, submit_time, cpu_req, gpu_req, base_duration):
        """
        Initialize the DigitalTwinTaskAgent.

        Args:
            task_id (str): Unique task identifier.
            submit_time (float): The time the task was submitted.
            cpu_req (float): CPU cores required.
            gpu_req (float): GPU cores required.
            base_duration (float): Base execution time without interference.
        """
        self.task_id = task_id
        self.submit_time = submit_time
        self.cpu_req = cpu_req
        self.gpu_req = gpu_req
        self.base_duration = base_duration

        # max_latency imposes a deadline for task freshness (1.5x base duration)
        self.max_latency = base_duration * 1.5

        # Valid states: 'WAITING', 'RUNNING', 'COMPLETED', 'TIMEOUT'
        self.state = 'WAITING'

        self.start_time = None
        self.end_time = None

class InterferenceModel:
    """
    Calculates the performance degradation (slowdown factor) due to background load.
    """
    @staticmethod
    def get_slowdown_factor(cluster_state_index):
        """
        Determine the task slowdown factor based on the cluster's current load state.

        Args:
            cluster_state_index (int): Index of the background load state.
                - 2 (High): Returns 1.2 (20% slowdown)
                - 3 (Overload): Returns 1.5 (50% slowdown)
                - Other: Returns 1.0 (no slowdown)

        Returns:
            float: The calculated slowdown factor.
        """
        if cluster_state_index == 2:
            return 1.5
        elif cluster_state_index == 3:
            return 2.0
        else:
            return 1.0
