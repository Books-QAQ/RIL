import os
import matplotlib.pyplot as plt

# ====== 输出文件夹（改这里）======
out_dir = r"/path/to/output_folder"
os.makedirs(out_dir, exist_ok=True)

# ========= 数据 =========
datasets = {
    "CORe50": {
        "x":  [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        "acc":[85.86, 87.28, 86.53, 85.54, 86.49, 85.27, 86.38, 85.37, 82.62, 82.63, 82.38],
        "fr": [ 1.03,  0.73,  0.72,  1.03,  0.41,  0.36,  0.16,  0.96,  1.03,  1.15,  0.00],
    },
    "iDigits": {
        "x":  [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        "acc":[22.83, 27.64, 24.77, 28.20, 34.84, 30.53, 36.69, 38.74, 38.74, 29.38, 29.38],
        "fr": [73.22, 64.43, 62.92, 54.63, 50.60, 58.42, 54.14, 55.25, 55.25, 60.91, 60.91],
    },
    "DomainNet": {
        "x":  [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0],
        "acc":[50.50,51.70,50.99,52.09,53.30,53.93,55.72,56.94,58.52,59.17,59.54,60.77,60.26,61.19,61.02,61.64,61.37,61.93,62.27,62.43,62.93,62.58,63.49,63.01,62.81,62.92],
        "fr": [25.62,25.86,27.07,26.06,23.95,23.61,20.94,19.45,17.76,16.54,15.69,14.01,15.13,13.96,14.29,13.36,13.61,12.79,12.41,11.96,11.73,11.82,10.80,11.08,11.13,11.32],
    },
}

# ========= 每张图单独指定横轴分度（刻度）=========
# 你可以按需要修改成任意刻度值列表（必须是数值）
xticks_map = {
    "CORe50":   [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    "iDigits":  [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    "DomainNet":[0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2],
}

# （可选）如果你还想分别控制每张图的横轴范围，也可以加这个：
# xlim_map = {
#     "CORe50": (0, 2.0),
#     "iDigits": (0, 2.0),
#     "DomainNet": (0, 5.0),
# }

def expand_ylim_by_fraction(vmin, vmax, pmin, pmax):
    """
    让数据值 vmin/vmax 分别落在纵轴的 pmin/pmax 位置（0~1）。
    返回需要设置的 (ylim_low, ylim_high)。
    """
    if not (0 <= pmin < pmax <= 1):
        raise ValueError("pmin/pmax 必须满足 0 <= pmin < pmax <= 1")
    if pmax == pmin:
        raise ValueError("pmin 不能等于 pmax")

    d = (vmax - vmin) / (pmax - pmin)   # 纵轴总跨度
    ylow = vmin - pmin * d
    yhigh = ylow + d
    return ylow, yhigh

# ✅ 每张图分别指定：左右轴“目标值区间”以及“它们在纵轴上的比例位置”
# 格式：
# "数据集": {
#   "acc": {"values":(vmin,vmax), "pos":(pmin,pmax)},
#   "fr":  {"values":(vmin,vmax), "pos":(pmin,pmax)}
# }
axis_fraction_map = {
    "CORe50": {
        "acc": {"values": (80, 90),   "pos": (0.15, 1.00)},
        "fr":  {"values": (0, 1.5),   "pos": (0.15, 0.80)},
    },
    "iDigits": {
        "acc": {"values": (20, 45),   "pos": (0.15, 0.90)},
        "fr":  {"values": (45, 80),   "pos": (0.10, 0.70)},
    },
    "DomainNet": {
        "acc": {"values": (50, 70),   "pos": (0.10, 0.95)},
        "fr":  {"values": (10, 30),   "pos": (0.08, 0.80)},
    },
}

for name, d in datasets.items():
    fig, ax1 = plt.subplots(figsize=(8, 2.8))

    # 左轴：Accuracy（实线）
    ax1.plot(d["x"], d["acc"], linestyle='-', marker='o', label='Accuracy% ↑')
    ax1.set_xlabel('λ')
    ax1.set_ylabel('Accuracy')

    # 右轴：Forgetting Rate（虚线）
    ax2 = ax1.twinx()
    ax2.plot(d["x"], d["fr"], linestyle='--', marker='s', label='Forgetting% ↓')
    ax2.set_ylabel('Forgetting')

    ax1.set_title(name)

    # 横轴刻度（可选）
    if name in xticks_map:
        ax1.set_xticks(xticks_map[name])

    # ✅ 左右轴按“比例位置”反推 ylim
    cfg = axis_fraction_map.get(name, {})
    if "acc" in cfg:
        vmin, vmax = cfg["acc"]["values"]
        pmin, pmax = cfg["acc"]["pos"]
        ax1.set_ylim(*expand_ylim_by_fraction(vmin, vmax, pmin, pmax))

    if "fr" in cfg:
        vmin, vmax = cfg["fr"]["values"]
        pmin, pmax = cfg["fr"]["pos"]
        ax2.set_ylim(*expand_ylim_by_fraction(vmin, vmax, pmin, pmax))

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax1.grid(True, linestyle=':')
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{name}_acc_fr.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", save_path)
