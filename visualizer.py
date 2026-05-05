"""
Visualization module for Multi-Agent Digital Twin Simulation results.
Generates charts and metrics based on simulation output.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(results_csv="outputs/sim_results.csv", output_dir="outputs/plots"):
    """
    Reads the simulation results CSV and creates visual comparisons
    between the baseline and MAS schedulers.
    """
    if not os.path.exists(results_csv):
        print(f"Error: {results_csv} not found. Run simulation first.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} simulation records for visualization.")

    # 1. Status Distribution (COMPLETED vs TIMEOUT)
    plt.figure(figsize=(10, 6))
    status_counts = df.groupby(['scheduler_type', 'status']).size().unstack(fill_value=0)

    if not status_counts.empty:
        status_counts.plot(kind='bar', stacked=True, color=['#4CAF50', '#F44336'])
        plt.title('Task Status Distribution by Scheduler')
        plt.xlabel('Scheduler Type')
        plt.ylabel('Number of Tasks')
        plt.xticks(rotation=0)
        plt.legend(title='Status')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'status_distribution.png'))
        plt.close()
        print(f"Generated {output_dir}/status_distribution.png")

    # 2. Average Actual Duration Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='scheduler_type', y='actual_duration', data=df, errorbar=None, hue='scheduler_type', legend=False)
    plt.title('Average Actual Task Duration by Scheduler')
    plt.xlabel('Scheduler Type')
    plt.ylabel('Average Actual Duration (Ticks)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_duration.png'))
    plt.close()
    print(f"Generated {output_dir}/avg_duration.png")

    # 3. Task Distribution Across Clusters
    plt.figure(figsize=(12, 6))
    cluster_dist = df.groupby(['scheduler_type', 'cluster_id']).size().unstack(fill_value=0)

    if not cluster_dist.empty:
        cluster_dist.plot(kind='bar', colormap='viridis')
        plt.title('Task Distribution Across Clusters')
        plt.xlabel('Scheduler Type')
        plt.ylabel('Number of Tasks Assigned')
        plt.xticks(rotation=0)
        plt.legend(title='Cluster ID')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'))
        plt.close()
        print(f"Generated {output_dir}/cluster_distribution.png")

    print("\nVisualization generation complete!")

if __name__ == "__main__":
    create_visualizations()
