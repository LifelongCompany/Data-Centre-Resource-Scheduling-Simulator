import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    # Set seaborn style for academic look
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Load data
    df = pd.read_csv("outputs/sim_results.csv")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Only COMPLETED tasks for duration plots
    completed_df = df[df['status'] == 'COMPLETED']

    # 1. Delay Distribution (Violin Plot)
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=completed_df,
        x="scheduler_type",
        y="actual_duration",
        hue="scheduler_type",
        legend=False,
        palette="muted",
        inner="quartile"
    )
    plt.title("Task Execution Delay Distribution")
    plt.xlabel("Scheduler Type")
    plt.ylabel("Actual Duration")
    plt.tight_layout()
    plt.savefig("outputs/delay_distribution.png", dpi=300)
    plt.close()

    # 2. Timeout Rate (Bar Chart)
    # Calculate timeout rate per scheduler
    total_tasks = df.groupby('scheduler_type').size()
    timeout_tasks = df[df['status'] == 'TIMEOUT'].groupby('scheduler_type').size()

    # Handle case where there are no timeouts
    if timeout_tasks.empty:
        timeout_rates = pd.Series(0, index=total_tasks.index)
    else:
        timeout_rates = (timeout_tasks / total_tasks * 100).fillna(0)

    # Create DataFrame for plotting
    timeout_df = timeout_rates.reset_index(name='timeout_rate')

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=timeout_df,
        x="scheduler_type",
        y="timeout_rate",
        hue="scheduler_type",
        legend=False,
        palette="pastel"
    )
    plt.title("Task Timeout Rate")
    plt.xlabel("Scheduler Type")
    plt.ylabel("Timeout Rate (%)")

    # Add percentage labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("outputs/timeout_rate.png", dpi=300)
    plt.close()

    # 3. Timeline Stability (Scatter Plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=completed_df,
        x="submit_time",
        y="actual_duration",
        hue="scheduler_type",
        palette="deep",
        alpha=0.6,
        s=30,
        edgecolor=None
    )
    plt.title("Timeline Stability under Fluctuation")
    plt.xlabel("Submit Time")
    plt.ylabel("Actual Duration")
    plt.legend(title="Scheduler Type")
    plt.tight_layout()
    plt.savefig("outputs/timeline_scatter.png", dpi=300)
    plt.close()

    print("Successfully generated all plots in outputs/ directory.")

if __name__ == "__main__":
    generate_plots()
