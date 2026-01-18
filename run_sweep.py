import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 1. 设置要搜索的 sensitivity 范围 (起始, 结束, 步长)
# 注意：np.arange 的结束值是不包含在内的，所以写 1.1 才能包含 1.0
sensitivity_list = np.arange(5.0, 5.1, 0.2)

# 2. 设置通用的训练参数
# 请根据你的实际情况修改 data-path 和 dataset
# base_args = [
#     "--dataset", "iDigits",  # 修改为你的数据集路径
#     "--num_tasks", "10",
#     "--n_tasks","10",# 为了快速验证，可以先设小一点，或者设为你需要的完整轮数
#     "--batch-size", "24",
#     "--versatile_inc",
#     "--random_inc",
#     "--shuffle", "True",
#     "--seed", "42",
#     "--IC",
#     "--thre", "0.0",
#     "--beta", "0.01",
#     "--use_cast_loss",
#     "--k", "2"
#
# ]
#
# # 3. 实验结果保存的总目录
# BASE_OUTPUT_DIR = "/root/VIL_main13/iDigits2"

base_args = [
    "--dataset", "DomainNet",  # 修改为你的数据集路径
    "--num_tasks", "345",
    "--n_tasks","15",# 为了快速验证，可以先设小一点，或者设为你需要的完整轮数
    "--batch-size", "24",
    "--versatile_inc",
    "--random_inc",
    "--shuffle", "True",
    "--seed", "42",
    "--IC",
    "--thre", "0.0",
    "--beta", "0.01",
    "--use_cast_loss",
    "--k", "2",

]
BASE_OUTPUT_DIR = "/root/VIL_main13/DomanNet2"

# base_args = [
#     "--dataset", "CORe50",  # 修改为你的数据集路径
#     "--num_tasks", "50",
#     "--n_tasks","20",# 为了快速验证，可以先设小一点，或者设为你需要的完整轮数
#     "--batch-size", "24",
#     "--versatile_inc",
#     "--random_inc",
#     "--shuffle", "True",
#     "--seed", "42",
#     "--IC",
#     "--thre", "0.0",
#     "--beta", "0.01",
#     "--use_cast_loss",
#     "--k", "2",
#
# ]
# BASE_OUTPUT_DIR = "/root/VIL_main13/CORe502"


# ===========================================

def run_task(sensitivity):
    """运行单个任务"""
    s_val = f"{sensitivity:.2f}"

    # 为每个实验创建独立的输出文件夹
    exp_dir = os.path.join(BASE_OUTPUT_DIR, f"sens_{s_val}")

    print(f"\n[Runner] >>>>>>> Start executing: sensitivity = {s_val} <<<<<<<")
    print(f"[Runner] Saving to: {exp_dir}")

    # 构建完整的命令行命令
    # 相当于在终端执行: python main_all.py --sce_sensitivity 0.x --output_dir ... [base_args]
    cmd = [
              "python", "main_all.py",
              "--sce_sensitivity", s_val,
              "--output_dir", exp_dir
          ] + base_args

    # 调用系统命令执行
    try:
        # check=True 表示如果脚本报错(exit code != 0)，这里也会抛出异常停止，防止无效运行
        subprocess.run(cmd, check=True)
        print(f"[Runner] Finished sensitivity {s_val}.")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Task failed for sensitivity {s_val}. Error: {e}")


def main():
    # 创建总目录
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    print(f"Total experiments to run: {len(sensitivity_list)}")

    # 简单的循环执行
    for s in sensitivity_list:
        run_task(s)

    print("\n[Runner] All tasks completed.")


if __name__ == "__main__":
    main()