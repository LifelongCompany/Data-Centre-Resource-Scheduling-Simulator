# Multi-Agent Digital Twin Scheduling Simulator

This project is a data center digital twin scheduling simulator based on Markov chains and Multi-Agent Systems (MAS). It is designed to simulate task scheduling in cluster environments (such as Alibaba GPU clusters), accounting for real-time background load inference to optimize task placement and avoid system congestion.

## Overview

The simulator consists of several core components:
*   **Data Preparation (`data_sampler.py`)**: Cleans and samples real-world machine and task metric data to drive the simulation.
*   **Markov Environment (`markov_env.py`)**: Uses historical machine load data to build a transition probability matrix. It provides a `MarkovClusterGenerator` to step through and predict future background load states.
*   **Intelligent Agents (`agents.py`)**:
    *   `ClusterAgent`: Represents physical compute clusters, tracking total and available CPU/GPU resources.
    *   `DigitalTwinTaskAgent`: Represents computational tasks to be scheduled.
    *   `InterferenceModel`: Determines the task slowdown factor based on current background cluster load.
*   **Scheduling Logic (`scheduler.py`)**:
    *   `BaselineScheduler`: A greedy approach that selects the cluster with the most available resources.
    *   `MASTwinScheduler`: A predictive scheduler that leverages Markov chain probabilities to select clusters with the lowest future congestion risk.
*   **Main Simulation Engine (`main_simulation.py`)**: Uses `simpy` to run an event-driven simulation of the entire ecosystem. It manages background load cycles and task arrivals, and writes simulation logs and metrics to a unified output file.
*   **Visualization (`visualizer.py`)**: Processes the simulation output and generates insights and charts for easy comparison of the baseline and MAS schedulers.

## Prerequisites

Make sure to install the required Python libraries before running the project:

```bash
pip install pandas numpy simpy matplotlib seaborn
```

## Running the Simulator

1. **Prepare Data**:
   Ensure `data/sample_tasks.csv` and `data/sample_metrics.csv` are generated or placed in the `data/` directory. If you have the raw data, you can run the sampler:
   ```bash
   python data_sampler.py
   ```

2. **Run the Simulation**:
   Execute the main simulation script. This will run the simulation for both scheduling algorithms and output the unified results to `outputs/sim_results.csv`.
   ```bash
   python main_simulation.py
   ```

3. **Visualize Results**:
   Generate performance and distribution charts:
   ```bash
   python visualizer.py
   ```
   The plots will be saved to the `outputs/plots/` directory.

## Simulation Mechanics

*   **Background Load Iteration**: By default, the simulator updates cluster background load via a Markov stepping process. To align with the transition matrix timeline (5-minute intervals), the background state advances every 300 ticks in the SimPy engine.
*   **Congestion Modeling**: Tasks placed on highly loaded clusters will face dynamic interference delays. If a task exceeds its maximum allowed freshness latency constraints (`Wait_Time + Actual_Duration > Max_Latency`), it gets marked as `TIMEOUT`. Successful completions are marked `COMPLETED`.
