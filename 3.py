import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =========================
# 1) Avg.Acc 数据（不含±）
# =========================
data = {
    "iDigits": {
        "CIL": {"acc": [63.17, 69.97, 65.77, 71.53, 76.44]},
        "DIL": {"acc": [73.83, 77.42, 79.09, 84.83, 85.35]},
        "VIL": {"acc": [59.07, 63.30, 59.34, 75.11, 66.80]},
        "RIL": {"acc": [19.83, 17.83, 27.18, 24.63, 34.84]},
    },
    "CORe50": {
        "CIL": {"acc": [70.03, 77.85, 77.11, 80.85, 84.31]},
        "DIL": {"acc": [80.72, 84.36, 83.09, 89.01, 92.77]},
        "VIL": {"acc": [64.85, 69.28, 77.11, 83.18, 84.79]},
        "RIL": {"acc": [57.31, 61.50, 28.25, 80.22, 86.38]},
    },
    "DomainNet": {
        "CIL": {"acc": [60.90, 65.21, 65.06, 65.43, 65.34]},
        "DIL": {"acc": [48.55, 49.13, 44.67, 54.44, 56.63]},
        "VIL": {"acc": [48.98, 49.45, 49.01, 53.37, 54.21]},
        "RIL": {"acc": [37.45, 18.10, 54.09, 47.81, 63.49]},
    },
}

methods = ["L2P", "CODA-P", "LAE", "ICON", "UDDE(Ours)"]

# =========================
# 2) ✅ 场景改成 4 个：加上 RIL
# =========================
SCENARIOS = ["CIL", "DIL", "VIL", "RIL"]

METHODS_TO_PLOT = ["L2P", "CODA-P", "LAE", "ICON", "UDDE(Ours)"]

# =========================
# 3) 浅色系配色（可按喜好改）
# =========================
COLOR_MAP = {
    "L2P":        "#F28C8C",  # 浅粉（更深）
    "CODA-P":     "#F2A769",  # 浅橙（更深）
    "LAE":        "#F2D15A",  # 浅黄（更深）
    "ICON":       "#6FB6E6",  # 浅蓝（更深）
    "UDDE(Ours)": "#8FD17A",  # 浅绿（更深）
}

# =========================
# 4) 每张图 y轴终止比例
# =========================
YMAX_RATIO = {
    "iDigits":  1.20,
    "CORe50":   1.35,
    "DomainNet":1.40,
}
YTICK_NBINS = {
    "iDigits":  6,
    "CORe50":   6,
    "DomainNet":6,
}

# =========================
# 5) 画 1×3 分组柱状图（每张子图右上角都有图例）
# =========================
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)

for ax, (ds_name, ds) in zip(axes, data.items()):
    x = np.arange(len(SCENARIOS))        # 4个场景
    n_methods = len(METHODS_TO_PLOT)
    width = 0.82 / n_methods

    all_vals = []

    for j, m in enumerate(METHODS_TO_PLOT):
        vals = np.array([ds[s]["acc"][methods.index(m)] for s in SCENARIOS], dtype=float)
        all_vals.append(vals)

        ax.bar(
            x + (j - (n_methods - 1) / 2) * width,
            vals,
            width=width,
            label=m,
            color=COLOR_MAP.get(m, "#DDDDDD"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95
        )

    all_vals = np.concatenate(all_vals)
    ymax = float(np.max(all_vals)) * float(YMAX_RATIO.get(ds_name, 1.10))

    ax.set_title(ds_name)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Average Accuracy")
    ax.grid(axis="y", alpha=0.2)

    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=YTICK_NBINS.get(ds_name, 6)))

    # ✅ 每张子图右上角图例（颜色块）
    ax.legend(loc="upper right", frameon=True, framealpha=0.85, fontsize=9)

plt.show()
