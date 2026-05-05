import os
import pandas as pd
import numpy as np


# ==========================================
# 阶段一：数据获取与微缩采样 (终极指定表头版)
# 文件名：data_sampler.py
# ==========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[*] 已创建目录: {directory}")


def process_machine_metrics():
    raw_path = 'raw_data/pai_machine_metric.csv'
    out_path = 'data/sample_metrics.csv'

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"未找到原始数据: {raw_path}")

    print(f"[*] 正在读取机器负载数据: {raw_path}")

    # 终极修复：直接强行赋予从 header 文件中得知的确切列名[cite: 1]
    metric_columns = [
        'worker_name', 'machine', 'start_time', 'end_time',
        'machine_cpu_iowait', 'machine_cpu_kernel', 'machine_cpu_usr',
        'machine_gpu', 'machine_load_1', 'machine_net_receive',
        'machine_num_worker', 'machine_cpu'
    ]

    # header=None 告诉 pandas 原文件没有表头，不要吃掉第一行数据
    df = pd.read_csv(raw_path, nrows=1000000, header=None, names=metric_columns)

    # 1. 重命名所需的列
    rename_dict = {
        'machine': 'machine_sn',
        'start_time': 'timestamp',
        'machine_cpu': 'cpu_usage',
        'machine_gpu': 'gpu_usage'
    }
    df = df[['machine', 'start_time', 'machine_cpu', 'machine_gpu']].rename(columns=rename_dict)

    # 2. 异常值处理与量纲转换
    # 删除没有使用率的废弃数据行
    df = df.dropna(subset=['cpu_usage', 'gpu_usage'], how='all')
    df['cpu_usage'] = df['cpu_usage'].fillna(0.0)
    df['gpu_usage'] = df['gpu_usage'].fillna(0.0)

    # 强制量纲归一化到 0-100% (处理阿里多卡累加数据)
    max_cpu = df['cpu_usage'].max()
    if max_cpu > 105.0:
        df['cpu_usage'] = (df['cpu_usage'] / max_cpu) * 100.0

    max_gpu = df['gpu_usage'].max()
    if max_gpu > 105.0:
        df['gpu_usage'] = (df['gpu_usage'] / max_gpu) * 100.0

    # 3. 筛选活跃机器
    active_machines = df.groupby('machine_sn').filter(
        lambda x: x['cpu_usage'].max() > 10.0 and len(x) > 50
    )['machine_sn'].unique()

    if len(active_machines) == 0:
        print("[!] 警告：未找到满足峰值>10%的活跃机器，放宽筛选条件...")
        active_machines = df['machine_sn'].unique()

    sample_size = min(100, len(active_machines))
    print(f"[*] 从 {len(active_machines)} 台机器中抽取 {sample_size} 台...")
    selected_machines = np.random.choice(active_machines, size=sample_size, replace=False)

    # 截取前 288 个时间步
    final_df = df[df['machine_sn'].isin(selected_machines)].copy()
    final_df = final_df.groupby('machine_sn').head(288)

    final_df.to_csv(out_path, index=False)
    print(f"[*] 机器指标数据已保存 -> {out_path} ({len(final_df)} 行)")


def process_task_table():
    raw_path = 'raw_data/pai_task_table.csv'
    out_path = 'data/sample_tasks.csv'

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"未找到原始数据: {raw_path}")

    print(f"\n[*] 正在读取任务调度数据: {raw_path}")

    # 终极修复：赋予已知确切的任务表头[cite: 2]
    task_columns = [
        'job_name', 'task_name', 'inst_num', 'status',
        'start_time', 'end_time', 'plan_cpu', 'plan_mem',
        'plan_gpu', 'gpu_type'
    ]

    df = pd.read_csv(raw_path, nrows=500000, header=None, names=task_columns)

    # 1. 提取并重命名列
    rename_map = {
        'job_name': 'task_id',
        'start_time': 'submit_time',
        'plan_cpu': 'cpu_req',
        'plan_gpu': 'gpu_req'
    }

    df = df[['job_name', 'start_time', 'end_time', 'plan_cpu', 'plan_gpu']].rename(columns=rename_map)

    # 2. 计算 duration
    df['duration'] = df['end_time'] - df['submit_time']

    # 3. 清洗数据
    df = df.dropna(subset=['submit_time', 'duration'])
    df = df[df['duration'] > 0]

    # 填补资源请求，空缺则填默认值
    df['cpu_req'] = df['cpu_req'].fillna(1.0)
    df['gpu_req'] = df['gpu_req'].fillna(0.0)

    # 4. 采样
    sample_size = min(1000, len(df))
    final_tasks = df.sample(n=sample_size, random_state=42).sort_values('submit_time')
    final_tasks.to_csv(out_path, index=False)
    print(f"[*] 任务样本数据已保存 -> {out_path} ({len(final_tasks)} 行)")


if __name__ == "__main__":
    print("=== 开始执行阶段一：真实数据清洗与微缩采样 ===")
    try:
        ensure_dir('data')
        process_machine_metrics()
        process_task_table()
        print("\n=== 阶段一执行完毕！后续系统组件可直接调用！ ===")
    except Exception as e:
        import traceback

        print(f"\n[!] 脚本执行遇到严重错误:")
        traceback.print_exc()