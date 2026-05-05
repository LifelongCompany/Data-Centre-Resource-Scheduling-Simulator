import markov_env

class ClusterAgent:
    """
    集群切片智能体，代表聚合后的资源池。
    管理总资源、可用资源，并包含背景负载生成器。
    """
    def __init__(self, cluster_id, bg_load_generator, cpu_total=10000, gpu_total=10000):
        """
        初始化 ClusterAgent。

        参数:
        - cluster_id: 集群ID
        - bg_load_generator: markov_env.MarkovClusterGenerator 实例，用于预测背景负载状态
        - cpu_total: 总CPU核数（默认10000）
        - gpu_total: 总GPU核数（默认10000）
        """
        self.cluster_id = cluster_id
        self.cpu_total = cpu_total
        self.gpu_total = gpu_total

        # 初始时，可用资源等于总资源
        self.cpu_available = cpu_total
        self.gpu_available = gpu_total

        # 内部持有 MarkovClusterGenerator 实例
        self.bg_load_generator = bg_load_generator

    def can_accept(self, task):
        """
        判断当前可用资源是否满足任务需求。

        参数:
        - task: DigitalTwinTaskAgent 实例，代表待分配的任务

        返回:
        - True 如果可用资源足够，否则 False
        """
        return (self.cpu_available >= task.cpu_req) and (self.gpu_available >= task.gpu_req)

    def allocate(self, task):
        """
        分配资源给任务，从当前可用资源中扣除任务所需的资源。
        调用前建议先通过 can_accept 检查资源是否充足。

        参数:
        - task: DigitalTwinTaskAgent 实例
        """
        if self.can_accept(task):
            self.cpu_available -= task.cpu_req
            self.gpu_available -= task.gpu_req
        else:
            raise ValueError(f"Cluster {self.cluster_id} 资源不足，无法分配给任务 {task.task_id}")

    def release(self, task):
        """
        任务完成时归还资源，将任务所占用的资源加回当前可用资源中。

        参数:
        - task: DigitalTwinTaskAgent 实例
        """
        self.cpu_available += task.cpu_req
        self.gpu_available += task.gpu_req

        # 防御性编程，确保释放资源后不超过总资源限制
        if self.cpu_available > self.cpu_total:
            self.cpu_available = self.cpu_total
        if self.gpu_available > self.gpu_total:
            self.gpu_available = self.gpu_total

    def step_background_load(self):
        """
        更新并返回当前背景负载状态。

        返回:
        - 生成的背景负载状态索引
        """
        return self.bg_load_generator.step()

class DigitalTwinTaskAgent:
    """
    数字孪生任务智能体，代表系统中的每一个计算任务。
    包含任务的基本需求、时间属性以及执行状态。
    """
    def __init__(self, task_id, submit_time, cpu_req, gpu_req, base_duration):
        """
        初始化 DigitalTwinTaskAgent。

        参数:
        - task_id: 任务唯一标识
        - submit_time: 任务提交时间
        - cpu_req: 任务所需 CPU 资源
        - gpu_req: 任务所需 GPU 资源
        - base_duration: 基础执行时长
        """
        self.task_id = task_id
        self.submit_time = submit_time
        self.cpu_req = cpu_req
        self.gpu_req = gpu_req
        self.base_duration = base_duration

        # max_latency 代表虚实同步新鲜度约束，设为 base_duration 的 1.5 倍
        self.max_latency = base_duration * 1.5

        # 初始状态设为 'WAITING'
        # 可选状态: 'WAITING', 'RUNNING', 'COMPLETED', 'TIMEOUT'
        self.state = 'WAITING'

        # 初始时间戳均设为 None
        self.start_time = None
        self.end_time = None

class InterferenceModel:
    """
    干扰计算器，用于根据背景负载状态计算任务执行的变慢系数。
    """
    @staticmethod
    def get_slowdown_factor(cluster_state_index):
        """
        传入 Cluster 的背景状态索引，返回任务变慢系数。

        参数:
        - cluster_state_index: 背景负载状态索引
          - 2 (High): 返回 1.2（变慢 20%）
          - 3 (Overload): 返回 1.5（变慢 50%）
          - 其他: 返回 1.0（正常速度）

        返回:
        - 变慢系数值 (float)
        """
        if cluster_state_index == 2:
            return 1.2
        elif cluster_state_index == 3:
            return 1.5
        else:
            return 1.0
