# Multi-Agent Digital Twin Scheduling Simulator

## Abstract
This project presents a robust Digital Twin scheduling simulator for large-scale GPU/CPU cluster environments. Driven by a Multi-Agent System (MAS) architecture, the simulator dynamically mirrors production workload fluctuations. By mapping historical background telemetry onto a discrete Markov chain framework, the system forecasts interference patterns and applies congestion-aware placement logic, thereby significantly mitigating performance degradation and peak-hour task timeouts.

## System Architecture
The simulator is driven by five core components natively coupled to mimic real-world interactions:
*   **Data Sampler (`data_sampler.py`)**: Sanitizes and standardizes real-world data traces to form the basis of background load distributions.
*   **Markov Environment (`markov_env.py`)**: Generates an empirical state transition matrix from time-series metric data to predict granular workload behaviors across distinct clusters.
*   **Agent Abstraction (`agents.py`)**:
    *   `ClusterAgent`: Encapsulates independent hardware clusters managing isolated resource pools and stochastic background load dynamics.
    *   `DigitalTwinTaskAgent`: Embeds resource vectors and temporal constraints mapping to specific scheduling payloads.
    *   `InterferenceModel`: Mathematically imposes performance degradation proportional to real-time cluster congestion.
*   **Scheduling Controllers (`scheduler.py`)**:
    *   `BaselineScheduler`: Executes a standard capacity-first heuristic (LeastAllocated).
    *   `MASTwinScheduler`: Implements predictive look-ahead routing based on minimum projected interference likelihoods to preemptively circumvent "noisy neighbor" scenarios.
*   **Simulation Engine (`main_simulation.py`)**: A SimPy-based discrete event engine orchestrating asynchronous task arrivals and cyclic Markov state transitions, yielding a unified telemetry log of all scheduled execution traits.

## Mathematical Model

### State Transition Matrix ($P$)
The background load distribution is quantized into four operational states ($N=4$): Normal (0, 1), High (2), and Overload (3). The transition probabilities between states $i$ and $j$ form the Markov transition matrix $P$:

$$
P =
\begin{pmatrix}
p_{00} & p_{01} & p_{02} & p_{03} \\
p_{10} & p_{11} & p_{12} & p_{13} \\
p_{20} & p_{21} & p_{22} & p_{23} \\
p_{30} & p_{31} & p_{32} & p_{33}
\end{pmatrix}
$$

### Interference Factor (Slowdown Factor)
Task execution duration diverges from base allocation dependent on real-time node saturation, modeled by a deterministic Slowdown Factor ($S$). Total execution latency $T_{actual}$ is formulated as:

$$
T_{actual} = T_{base} \times S(State)
$$
Where:
- $S(Normal) = 1.0$
- $S(High) = 1.2$
- $S(Overload) = 1.5$

## Usage Guide

1.  **Environment Preparation**:
    Install the required dependencies to run the simulation and visualizations.
    ```bash
    pip install pandas numpy simpy matplotlib seaborn
    ```

2.  **Dataset Construction**:
    Generate the `sample_tasks.csv` and `sample_metrics.csv` files required to drive the state predictors.
    ```bash
    python data_sampler.py
    ```

3.  **Execute the Simulation**:
    Run the SimPy orchestrator. Results will be unified into `outputs/sim_results.csv`.
    ```bash
    python main_simulation.py
    ```

4.  **Visualize Telemetry**:
    Render comparative analytics charts to the `outputs/` directory.
    ```bash
    python visualizer.py
    ```

## Experimental Results

The simulation validates the superiority of the MAS architecture over traditional baseline heuristics.

### Delay Distribution
By evading high-congestion zones, the MAS scheduler compresses the tail distribution of execution latencies, demonstrating robust performance even under extreme node duress.

![Delay Distribution](outputs/delay_distribution.png)

### Timeout Rate
Congestion-aware routing significantly reduces failure occurrences, enforcing hard constraints on maximal latency thresholds ($T_{max} < T_{wait} + T_{actual}$).

![Timeout Rate](outputs/timeout_rate.png)

### Timeline Stability
The MAS algorithm actively shaves structural load peaks, as highlighted by a tightly clustered trajectory compared to the volatile baseline variances during concurrent submission spikes.

![Timeline Stability](outputs/timeline_scatter.png)
