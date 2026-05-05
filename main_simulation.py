"""
Main simulation script for Multi-Agent Digital Twin Scheduler.
Uses simpy for event-driven simulation and pandas for data handling.
"""

import os
import simpy
import pandas as pd
from markov_env import compute_transition_matrix, MarkovClusterGenerator
from agents import ClusterAgent, DigitalTwinTaskAgent, InterferenceModel
from scheduler import BaselineScheduler, MASTwinScheduler

# Schema mapping for dynamic dataset column handling
SCHEMA_CONFIG = {
    'task_id': 'task_id',
    'submit_time': 'submit_time',
    'cpu_req': 'cpu_req',
    'gpu_req': 'gpu_req',
    'duration': 'duration'
}

BOTTLENECK_CPU = 25000
BOTTLENECK_GPU = 25000

def setup_outputs_dir():
    os.makedirs("outputs", exist_ok=True)

def background_load_process(env, clusters):
    """
    Process 1: Background Load cycle.
    Steps the Markov background load state every 300 ticks (5 minutes).
    """
    step_size = 300
    while True:
        for cluster in clusters:
            cluster.step_background_load()
        yield env.timeout(step_size)

def task_arrival_process(env, tasks_df, clusters, scheduler, results_list, scheduler_type):
    """
    Process 2: Task Arrival handling.
    Dispatches tasks as per their submit_time and initiates task execution.
    """
    tasks_df_sorted = tasks_df.sort_values(by=SCHEMA_CONFIG['submit_time'])
    last_submit_time = tasks_df_sorted.iloc[0][SCHEMA_CONFIG['submit_time']]

    yield env.timeout(0)

    for _, row in tasks_df_sorted.iterrows():
        submit_time = row[SCHEMA_CONFIG['submit_time']]
        delay = submit_time - last_submit_time
        if delay > 0:
            yield env.timeout(delay)
        last_submit_time = submit_time

        task = DigitalTwinTaskAgent(
            task_id=row[SCHEMA_CONFIG['task_id']],
            submit_time=submit_time,
            cpu_req=row[SCHEMA_CONFIG['cpu_req']],
            gpu_req=row[SCHEMA_CONFIG['gpu_req']],
            base_duration=row[SCHEMA_CONFIG['duration']]
        )
        task.sim_arrival_time = env.now

        env.process(task_execution_process(env, task, clusters, scheduler, results_list, scheduler_type))

def task_execution_process(env, task, clusters, scheduler, results_list, scheduler_type):
    """
    Process 3: Task Execution lifecycle.
    Evaluates placement, allocates resources, computes load interference, and evaluates timeout.
    """
    WAIT_STEP = 5
    assigned_cluster = None

    while True:
        assigned_cluster = scheduler.schedule(task)
        if assigned_cluster is not None:
            break

        yield env.timeout(WAIT_STEP)
        wait_time = env.now - task.sim_arrival_time

        if wait_time > task.max_latency:
            task.state = 'TIMEOUT'
            results_list.append({
                'task_id': task.task_id,
                'scheduler_type': scheduler_type,
                'cluster_id': None,
                'submit_time': task.submit_time,
                'wait_time': wait_time,
                'actual_duration': 0,
                'status': task.state
            })
            return

    assigned_cluster.allocate(task)
    task.start_time = env.now
    task.state = 'RUNNING'

    current_cluster_state = assigned_cluster.bg_load_generator.current_state
    slowdown_factor = InterferenceModel.get_slowdown_factor(current_cluster_state)
    actual_duration = task.base_duration * slowdown_factor
    wait_time = task.start_time - task.sim_arrival_time

    yield env.timeout(actual_duration)

    task.end_time = env.now

    if wait_time + actual_duration > task.max_latency:
        task.state = 'TIMEOUT'
    else:
        task.state = 'COMPLETED'

    assigned_cluster.release(task)

    results_list.append({
        'task_id': task.task_id,
        'scheduler_type': scheduler_type,
        'cluster_id': assigned_cluster.cluster_id,
        'submit_time': task.submit_time,
        'wait_time': wait_time,
        'actual_duration': actual_duration,
        'status': task.state
    })

def run_simulation(scheduler_type="baseline", sim_time=1440, task_limit=None):
    """
    Configures and runs the SimPy environment for a specified scheduler.
    """
    env = simpy.Environment()

    transition_matrix = compute_transition_matrix("data/sample_metrics.csv")

    clusters = []
    for i in range(4):
        bg_generator = MarkovClusterGenerator(transition_matrix)
        cluster = ClusterAgent(cluster_id=i, bg_load_generator=bg_generator, cpu_total=BOTTLENECK_CPU, gpu_total=BOTTLENECK_GPU)
        clusters.append(cluster)

    if scheduler_type == "baseline":
        scheduler = BaselineScheduler(clusters)
    else:
        scheduler = MASTwinScheduler(clusters)

    tasks_df = pd.read_csv("data/sample_tasks.csv")
    if task_limit is not None:
        tasks_df = tasks_df.head(task_limit)

    results_list = []

    env.process(background_load_process(env, clusters))
    env.process(task_arrival_process(env, tasks_df, clusters, scheduler, results_list, scheduler_type))

    env.run(until=sim_time)

    return results_list

if __name__ == "__main__":
    setup_outputs_dir()

    tasks_df = pd.read_csv("data/sample_tasks.csv")

    max_delay = tasks_df['submit_time'].max() - tasks_df['submit_time'].min()
    max_time = max_delay + tasks_df['duration'].max() * 2

    print("Running Baseline Simulation...")
    baseline_results = run_simulation(scheduler_type="baseline", sim_time=max_time)

    print("Running MAS Simulation...")
    mas_results = run_simulation(scheduler_type="mas", sim_time=max_time)

    all_results = baseline_results + mas_results
    results_df = pd.DataFrame(all_results)

    output_path = "outputs/sim_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Simulation completed. Results saved to {output_path}")
