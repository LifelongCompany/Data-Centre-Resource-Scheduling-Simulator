import pandas as pd
import numpy as np
import hashlib

def get_cluster_id(machine_sn):
    return int(hashlib.md5(str(machine_sn).encode()).hexdigest(), 16) % 4

def get_state(cpu_usage):
    if cpu_usage <= 20.0:
        return 0
    elif cpu_usage <= 60.0:
        return 1
    elif cpu_usage <= 85.0:
        return 2
    else:
        return 3

def compute_transition_matrix(csv_path="data/sample_metrics.csv"):
    df = pd.read_csv(csv_path)

    # Defensive check for cpu_usage
    max_cpu = df['cpu_usage'].max()
    if max_cpu > 0 and max_cpu <= 1.0:
        df['cpu_usage'] = df['cpu_usage'] * 100.0

    df['cluster_id'] = df['machine_sn'].apply(get_cluster_id)

    # Sort by timestamp to ensure chronological order
    df = df.sort_values(by=['timestamp'])

    # Group by cluster_id and timestamp to get mean cpu_usage
    agg_df = df.groupby(['cluster_id', 'timestamp'])['cpu_usage'].mean().reset_index()

    # Map mean cpu_usage to states
    agg_df['state'] = agg_df['cpu_usage'].apply(get_state)

    # Compute transition matrix
    num_states = 4
    transition_counts = np.zeros((num_states, num_states))

    for cluster_id in range(4):
        cluster_data = agg_df[agg_df['cluster_id'] == cluster_id].sort_values(by='timestamp')
        states = cluster_data['state'].values
        for i in range(len(states) - 1):
            curr_state = states[i]
            next_state = states[i+1]
            transition_counts[curr_state, next_state] += 1

    # Convert counts to probabilities
    transition_matrix = np.zeros((num_states, num_states))
    for i in range(num_states):
        row_sum = np.sum(transition_counts[i, :])
        if row_sum > 0:
            transition_matrix[i, :] = transition_counts[i, :] / row_sum
        else:
            # Self-loop for empty states
            transition_matrix[i, i] = 1.0

    return transition_matrix

class MarkovClusterGenerator:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
        self.current_state = 1  # Start at Normal state (index 1)

    def step(self):
        probs = self.transition_matrix[self.current_state]
        self.current_state = np.random.choice(4, p=probs)
        return self.current_state

    def predict_future_state(self, steps):
        # one-hot encoding of current state
        v = np.zeros(4)
        v[self.current_state] = 1.0

        # P^steps
        P_steps = np.linalg.matrix_power(self.transition_matrix, steps)

        # future probability distribution
        return np.dot(v, P_steps)

if __name__ == "__main__":
    P = compute_transition_matrix("data/sample_metrics.csv")
    print("Transition Matrix P:")
    print(np.round(P, 4))

    print("\nStarting Simulation:")
    generator = MarkovClusterGenerator(P)

    for i in range(1, 11):
        next_state = generator.step()
        future_prob = generator.predict_future_state(1)
        print(f"Step {i:2d} | Current State: {next_state} | Next Step Probs: {np.round(future_prob, 4)}")
