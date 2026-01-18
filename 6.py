import os, re, glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _extract_step_id_from_name(path):
    m = re.search(r"_step(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _parse_task_id_from_filename(path):
    m = re.search(r"(?:^|_)task(\d+)(?:_|\.|$)", os.path.basename(path), flags=re.IGNORECASE)
    return int(m.group(1)) - 1 if m else None  # 0-based


def load_adapter_vec_and_meta(p, device="cpu"):
    ckpt = torch.load(p, map_location=device)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    params = [v.view(-1) for k, v in sd.items() if "adapter" in k]
    if not params:
        raise ValueError(f"No adapter params in {p}")
    vec = torch.cat(params).to(device)

    meta = {}
    if isinstance(ckpt, dict):
        for k in ["task_id", "epoch", "step_in_epoch", "global_step"]:
            if k in ckpt:
                meta[k] = int(ckpt[k])

    if "task_id" not in meta:
        tid = _parse_task_id_from_filename(p)
        if tid is not None:
            meta["task_id"] = tid

    if "global_step" not in meta:
        sid = _extract_step_id_from_name(p)
        if sid is not None:
            meta["global_step"] = sid

    return vec, meta


def compute_updates(vecs):
    return [vecs[i] - vecs[i - 1] for i in range(1, len(vecs))]


def compute_sims_from_updates(updates):
    sims = []
    for i in range(len(updates) - 1):
        sims.append(F.cosine_similarity(updates[i].unsqueeze(0), updates[i + 1].unsqueeze(0)).item())
    return sims


def debug_task(task_id, vecs):
    M = len(vecs)
    if M < 3:
        print(f"[T{task_id+1}] M={M} (<3) skip")
        return

    updates = [vecs[i] - vecs[i - 1] for i in range(1, M)]
    norms = torch.tensor([u.norm().item() for u in updates])

    sims = []
    for i in range(len(updates) - 1):
        s = F.cosine_similarity(updates[i].unsqueeze(0), updates[i + 1].unsqueeze(0)).item()
        sims.append(s)

    sims_np = np.array(sims, dtype=np.float64)
    print(
        f"[T{task_id+1}] M={M}, K={len(sims)} | "
        f"update_norm min={norms.min():.3e}, max={norms.max():.3e} | "
        f"sims mean={sims_np.mean():.6f}, min={sims_np.min():.6f}, max={sims_np.max():.6f}"
    )
    print(f"    sims={np.array2string(sims_np, precision=6)}")


def per_task_sims_and_strength(output_dir, device="cpu", epoch1_only=True, epoch1_index=0, do_debug=True):
    """
    Returns dict tid -> {
        'K': int,
        'sims': np.ndarray [K],
        'strength_mean'/'min'/'median'
    }
    """
    step_dir = os.path.join(output_dir, "checkpoint_step")
    paths = sorted(glob.glob(os.path.join(step_dir, "*.pth")))
    if not paths:
        raise FileNotFoundError(f"No step ckpts under {step_dir}")

    # group by task
    task_items = {}
    for p in paths:
        vec, meta = load_adapter_vec_and_meta(p, device=device)
        tid = meta.get("task_id", None)
        if tid is None:
            raise RuntimeError(f"Cannot determine task_id for {p} (need meta task_id or filename taskX)")

        if epoch1_only:
            ep = meta.get("epoch", None)
            if ep is not None and ep != epoch1_index:
                continue

        key = meta.get("global_step", None)
        if key is None:
            key = _extract_step_id_from_name(p)
        if key is None:
            key = p

        task_items.setdefault(tid, []).append((key, vec))

    out = {}
    for tid, items in task_items.items():
        items.sort(key=lambda x: x[0])
        vecs = [v for _, v in items]

        if do_debug:
            debug_task(tid, vecs)

        if len(vecs) < 3:
            continue

        upd = compute_updates(vecs)
        sims = np.asarray(compute_sims_from_updates(upd), dtype=np.float64)
        K = sims.size
        if K <= 0:
            continue

        out[tid] = {
            "K": int(K),
            "sims": sims,  # ✅ 保存下来，后面用来算 tau & conflict_rate
            "strength_mean": float(sims.mean()),
            "strength_min": float(sims.min()),
            "strength_median": float(np.median(sims)),
        }
    return out


def compute_global_tau_from_vil_quantile(vil_stats, common_tasks, q=0.2):
    """
    tau = quantile of all VIL sims pooled over common_tasks.
    """
    all_sims = []
    for t in common_tasks:
        all_sims.extend(vil_stats[t]["sims"].tolist())
    all_sims = np.asarray(all_sims, dtype=np.float64)
    if all_sims.size == 0:
        raise ValueError("No VIL sims available to compute tau.")
    tau = float(np.quantile(all_sims, q))
    return tau


def add_conflict_metrics(stats, tau, a=0.05):
    """
    Add:
      conflict_rate = mean(sim < tau)
      stealth_ratio = mean(-a <= sim < 0)
    """
    for t, d in stats.items():
        sims = d["sims"]
        d["conflict_rate"] = float((sims < tau).mean())
        d["stealth_ratio"] = float(((sims < 0.0) & (sims >= -a)).mean())
    return stats


def plot_per_task_bar(ril_stats, vil_stats, key="strength_mean", save_path="per_task_strength_mean.png"):
    common = sorted(set(ril_stats.keys()) & set(vil_stats.keys()))
    common = [t for t in common if ril_stats[t]["K"] > 0 and vil_stats[t]["K"] > 0]
    if not common:
        print("No common tasks with valid K.")
        return

    x = np.arange(len(common))
    ril_vals = np.array([ril_stats[t][key] for t in common], dtype=np.float64)
    vil_vals = np.array([vil_stats[t][key] for t in common], dtype=np.float64)

    width = 0.38
    plt.figure(figsize=(12, 4.8))
    plt.bar(x - width / 2, vil_vals, width=width, label="VIL")
    plt.bar(x + width / 2, ril_vals, width=width, label="RIL")

    plt.xticks(x, [f"T{t+1}" for t in common], rotation=0)
    plt.ylabel(key)
    plt.title(f"Per-task {key} (within-task step sims)")
    plt.grid(True, axis="y", linestyle=":", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")


def plot_conflict_rate_bar_with_global_avg(ril_stats, vil_stats, key="conflict_rate", save_path="per_task_conflict_rate.png"):
    common = sorted(set(ril_stats.keys()) & set(vil_stats.keys()))
    common = [t for t in common if ril_stats[t]["K"] > 0 and vil_stats[t]["K"] > 0]
    if not common:
        print("No common tasks with valid K.")
        return

    ril_vals = np.array([ril_stats[t][key] for t in common], dtype=np.float64)
    vil_vals = np.array([vil_stats[t][key] for t in common], dtype=np.float64)

    ril_K = np.array([ril_stats[t]["K"] for t in common], dtype=np.float64)
    vil_K = np.array([vil_stats[t]["K"] for t in common], dtype=np.float64)

    ril_global = float((ril_vals * ril_K).sum() / (ril_K.sum() + 1e-12))
    vil_global = float((vil_vals * vil_K).sum() / (vil_K.sum() + 1e-12))

    labels = [f"T{t+1}" for t in common] + ["GlobalAvg"]
    ril_vals = np.concatenate([ril_vals, [ril_global]])
    vil_vals = np.concatenate([vil_vals, [vil_global]])

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(12, 4.8))
    plt.bar(x - width / 2, vil_vals, width=width, label="VIL")
    plt.bar(x + width / 2, ril_vals, width=width, label="RIL")

    plt.xticks(x, labels, rotation=0)
    plt.ylabel(key)
    plt.title(f"Per-Task {key} (tau fixed from VIL quantile)")
    plt.grid(True, axis="y", linestyle=":", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")
    print(f"GlobalAvg {key} | VIL={vil_global:.6f}, RIL={ril_global:.6f}")


def plot_per_task_sims_boxplot(ril_stats, vil_stats, save_path="per_task_sims_boxplot.png", seed=0):
    """Per-task sims distribution (boxplot) + raw points, side-by-side for VIL vs RIL."""
    common = sorted(set(ril_stats.keys()) & set(vil_stats.keys()))
    common = [t for t in common if ril_stats[t]["K"] > 0 and vil_stats[t]["K"] > 0]
    if not common:
        print("No common tasks with valid K.")
        return

    vil_data = [np.asarray(vil_stats[t]["sims"], dtype=np.float64) for t in common]
    ril_data = [np.asarray(ril_stats[t]["sims"], dtype=np.float64) for t in common]

    x = np.arange(len(common), dtype=np.float64)
    offset = 0.18
    pos_vil = x - offset
    pos_ril = x + offset

    plt.figure(figsize=(max(12, 0.55 * len(common) + 4), 5.2))

    bp_vil = plt.boxplot(
        vil_data,
        positions=pos_vil,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    bp_ril = plt.boxplot(
        ril_data,
        positions=pos_ril,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )

    # Color boxes to match points (VIL=blue, RIL=orange)
    for b in bp_vil["boxes"]:
        b.set_alpha(0.35)
        b.set_facecolor("C0")
    for b in bp_ril["boxes"]:
        b.set_alpha(0.35)
        b.set_facecolor("C1")

    # ---- overlay raw points (with jitter) ----
    rng = np.random.default_rng(seed)
    jitter_scale = 0.06  # x-axis jitter amplitude

    for i in range(len(common)):
        # VIL points: blue
        yv = vil_data[i]
        xv = pos_vil[i] + rng.uniform(-jitter_scale, jitter_scale, size=yv.size)
        plt.scatter(
            xv, yv,
            s=10, alpha=0.70,
            c="C0", marker="o",
            linewidths=0
        )

        # RIL points: orange
        yr = ril_data[i]
        xr = pos_ril[i] + rng.uniform(-jitter_scale, jitter_scale, size=yr.size)
        plt.scatter(
            xr, yr,
            s=10, alpha=0.70,
            c="C1", marker="o",
            linewidths=0
        )

    plt.xticks(x, [f"T{t+1}" for t in common], rotation=0)
    plt.ylabel(r"Cosine Similarity for Gradient Updates")
    plt.title("")
    plt.grid(True, axis="y", linestyle=":", alpha=0.3)

    # Legend: box patches
    plt.legend([bp_vil["boxes"][0], bp_ril["boxes"][0]], ["VIL", "RIL"], loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")


def plot_per_task_sims_boxplot_3way(icon_ril_stats, icon_vil_stats, udde_ril_stats,
                                   save_path="per_task_sims_boxplot_3way.png",
                                   seed=0):
    """
    Per-task sims distribution (boxplot + raw points), 3 columns per task:
      ICON-VIL (blue), ICON-RIL (orange), UDDE-RIL (green)
    """
    # common tasks across all three
    common = sorted(set(icon_ril_stats.keys()) & set(icon_vil_stats.keys()) & set(udde_ril_stats.keys()))
    common = [
        t for t in common
        if icon_ril_stats[t]["K"] > 0 and icon_vil_stats[t]["K"] > 0 and udde_ril_stats[t]["K"] > 0
    ]
    if not common:
        print("No common tasks with valid K across ICON-RIL / ICON-VIL / UDDE-RIL.")
        return

    vil_data  = [np.asarray(icon_vil_stats[t]["sims"], dtype=np.float64) for t in common]
    iril_data = [np.asarray(icon_ril_stats[t]["sims"], dtype=np.float64) for t in common]
    uril_data = [np.asarray(udde_ril_stats[t]["sims"], dtype=np.float64) for t in common]

    x = np.arange(len(common), dtype=np.float64)

    # three columns
    offset = 0.26
    pos_vil  = x - offset
    pos_iril = x
    pos_uril = x + offset

    plt.figure(figsize=(max(12, 0.55 * len(common) + 4), 5.2))

    bp_vil = plt.boxplot(
        vil_data, positions=pos_vil, widths=0.22,
        patch_artist=True, showfliers=False, manage_ticks=False
    )
    bp_iril = plt.boxplot(
        iril_data, positions=pos_iril, widths=0.22,
        patch_artist=True, showfliers=False, manage_ticks=False
    )
    bp_uril = plt.boxplot(
        uril_data, positions=pos_uril, widths=0.22,
        patch_artist=True, showfliers=False, manage_ticks=False
    )

    # color boxes (match points)
    for b in bp_vil["boxes"]:
        b.set_alpha(0.35)
        b.set_facecolor("C0")   # blue
    for b in bp_iril["boxes"]:
        b.set_alpha(0.35)
        b.set_facecolor("C1")   # orange
    for b in bp_uril["boxes"]:
        b.set_alpha(0.35)
        b.set_facecolor("C2")   # green

    # overlay raw points with jitter
    rng = np.random.default_rng(seed)
    jitter_scale = 0.05

    for i in range(len(common)):
        # ICON-VIL (blue)
        y = vil_data[i]
        xx = pos_vil[i] + rng.uniform(-jitter_scale, jitter_scale, size=y.size)
        plt.scatter(xx, y, s=10, alpha=0.70, c="C0", marker="o", linewidths=0)

        # ICON-RIL (orange)
        y = iril_data[i]
        xx = pos_iril[i] + rng.uniform(-jitter_scale, jitter_scale, size=y.size)
        plt.scatter(xx, y, s=10, alpha=0.70, c="C1", marker="o", linewidths=0)

        # UDDE-RIL (green)
        y = uril_data[i]
        xx = pos_uril[i] + rng.uniform(-jitter_scale, jitter_scale, size=y.size)
        plt.scatter(xx, y, s=10, alpha=0.70, c="C2", marker="o", linewidths=0)

    # x-axis
    plt.xticks(x, [f"T{t+1}" for t in common], rotation=0)

    # y-axis label（你要公式的话用这一行）
    # plt.ylabel(r"$\cos\!\left(s_{i-1,i},\, s_{i,i+1}\right)$")
    plt.ylabel(r"Cosine Similarity for Gradient Updates")

    plt.title("")
    plt.grid(True, axis="y", linestyle=":", alpha=0.3)

    plt.legend(
        [bp_vil["boxes"][0], bp_iril["boxes"][0], bp_uril["boxes"][0]],
        ["ICON-VIL", "ICON-RIL", "UDDE-RIL"],
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    ICON_RIL_PATH = "/root/VIL_main13/output4/output_ril_iD_423"
    ICON_VIL_PATH = "/root/VIL_main13/output4/output_vil_iD_423"
    UDDE_RIL_PATH = "/root/VIL_main13/output4/output_ril_iD_422"

    # ===== settings =====
    epoch1_only = True
    epoch1_index = 0
    a = 0.05
    q = 0.2
    do_debug = True

    icon_ril = per_task_sims_and_strength(ICON_RIL_PATH, device="cpu",
                                          epoch1_only=epoch1_only, epoch1_index=epoch1_index, do_debug=do_debug)
    icon_vil = per_task_sims_and_strength(ICON_VIL_PATH, device="cpu",
                                          epoch1_only=epoch1_only, epoch1_index=epoch1_index, do_debug=do_debug)
    udde_ril = per_task_sims_and_strength(UDDE_RIL_PATH, device="cpu",
                                          epoch1_only=epoch1_only, epoch1_index=epoch1_index, do_debug=do_debug)

    # 只用 ICON(RIL,VIL) 的共同任务来算 tau（保持你原来的“VIL做参考”的逻辑）
    common_tasks = sorted(set(icon_ril.keys()) & set(icon_vil.keys()))
    common_tasks = [t for t in common_tasks if icon_ril[t]["K"] > 0 and icon_vil[t]["K"] > 0]
    if not common_tasks:
        raise RuntimeError("No common tasks with valid K to compute tau.")

    tau = compute_global_tau_from_vil_quantile(icon_vil, common_tasks, q=q)
    print(f"\n=== Global tau from VIL quantile q={q:.2f}: tau={tau:.6f} ===\n")

    # 用同一个 tau 计算三者的 conflict_rate（可选，你原来会画 conflict_rate 图的话还可继续）
    icon_ril = add_conflict_metrics(icon_ril, tau=tau, a=a)
    icon_vil = add_conflict_metrics(icon_vil, tau=tau, a=a)
    udde_ril = add_conflict_metrics(udde_ril, tau=tau, a=a)

    # 图1：conflict_rate（仍然是两路函数；你要三路柱状也可以再加）
    plot_conflict_rate_bar_with_global_avg(icon_ril, icon_vil, key="conflict_rate",
                                           save_path="Do_ICON_per_task_conflict_rate.png")

    # 图2：三路箱式图（每个 task 3 列）
    plot_per_task_sims_boxplot_3way(icon_ril, icon_vil, udde_ril,
                                    save_path="Do_per_task_sims_boxplot_3way.png",
                                    seed=0)
