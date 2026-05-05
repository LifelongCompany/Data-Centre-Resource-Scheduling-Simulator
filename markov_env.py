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

    # Normalize CPU usage if bounded between 0 and 1
    max_cpu = df['cpu_usage'].max()
    if max_cpu > 0 and max_cpu <= 1.0:
        df['cpu_usage'] = df['cpu_usage'] * 100.0

    df['cluster_id'] = df['machine_sn'].apply(get_cluster_id)

    # Ensure chronological order for transition calculations
    df = df.sort_values(by=['timestamp'])

    # Aggregate mean cpu_usage by cluster_id and timestamp
    agg_df = df.groupby(['cluster_id', 'timestamp'])['cpu_usage'].mean().reset_index()

    agg_df['state'] = agg_df['cpu_usage'].apply(get_state)

    num_states = 4
    transition_counts = np.zeros((num_states, num_states))

    for cluster_id in range(4):
        cluster_data = agg_df[agg_df['cluster_id'] == cluster_id].sort_values(by='timestamp')
        states = cluster_data['state'].values
        for i in range(len(states) - 1):
            curr_state = states[i]
            next_state = states[i+1]
            transition_counts[curr_state, next_state] += 1

    transition_matrix = np.zeros((num_states, num_states))
    for i in range(num_states):
        row_sum = np.sum(transition_counts[i, :])
        if row_sum > 0:
            transition_matrix[i, :] = transition_counts[i, :] / row_sum
        else:
            transition_matrix[i, i] = 1.0

    return transition_matrix

class MarkovClusterGenerator:
    """
    Simulates background load transitions for a cluster based on a given Markov transition matrix.
    """
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
        # Default starting state is Normal (index 1)
        self.current_state = 1

    def step(self):
        """
        Advance the background load to the next state based on the transition probabilities.
        """
        probs = self.transition_matrix[self.current_state]
        self.current_state = np.random.choice(4, p=probs)
        return self.current_state

    def predict_future_state(self, steps):
        """
        Predict the probability distribution of future states after a given number of steps.
        """
        v = np.zeros(4)
        v[self.current_state] = 1.0
        P_steps = np.linalg.matrix_power(self.transition_matrix, steps)
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
