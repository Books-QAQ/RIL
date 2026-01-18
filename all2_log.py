import math
import sys
import os
import datetime
import json
from turtle import undo
from typing import Iterable
from pathlib import Path
import copy
from collections import deque
import torch
import numpy as np
from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.utils.model_ema import ModelEmaV2
import utils
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
class Engine():
    def __init__(self, model=None, device=None, class_mask=[], domain_list=[], args=None):
        self.current_task = 0
        self.current_classes = []
        self.class_group_num = 5
        self.classifier_pool = [None for _ in range(self.class_group_num)]
        self.class_group_train_count = [0 for _ in range(self.class_group_num)]

        self.task_num = len(class_mask)
        self.class_group_size = len(class_mask[0])
        self.distill_head = None
        self.model = model
        self.num_classes = max([item for mask in class_mask for item in mask]) + 1
        self.labels_in_head = np.arange(self.num_classes)
        self.added_classes_in_cur_task = set()
        self.head_timestamps = np.zeros_like(self.labels_in_head)
        self.args = args
        self.class_mask = class_mask
        self.domain_list = domain_list
        self.task_type = "initial"
        self.args = args
        self.adapter_vec = []
        self.task_type_list = []
        self.class_group_list = []
        self.adapter_vec_label = []
        self.device = device
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((self.args.class_num, self.args.domain_num))
            self.label_train_count = np.zeros((self.args.class_num))
            self.tanh = torch.nn.Tanh()
            self.last_eval_class_acc = None

        self.cs = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # ==== Episodic Replay (memory & schedule) ====
        self.use_replay = getattr(self.args, 'use_replay', False)
        self.replay_size = int(getattr(self.args, 'replay_size', 2000))
        self.replay_ratio = float(getattr(self.args, 'replay_ratio', 0.25))
        self.rng = np.random.RandomState(getattr(self.args, 'replay_seed', 0))

        # class -> list[(x_cpu, y_cpu)]
        self.replay_buffer = {}
        self.class_seen = np.zeros(self.num_classes, dtype=np.int64)
        self.cum_counts_total = np.zeros(self.num_classes, dtype=np.int64)
        self.loss_weight = None  # 类级权重，shape=(C,)
        self.pair_loss_weight = None  # 类×域 组级权重，shape=(C, D)
        self.domain_loss_weight = None  # 域级权重，shape=(D,)
        self.use_forget_priority = bool(getattr(self.args, 'replay_forget', False))
        self.forget_weight = float(getattr(self.args, 'forget_weight', 0.9))  # 与反频率混合权重
        self.forget_alpha = float(getattr(self.args, 'forget_alpha', 0.3))  # 每类遗忘分数的 EMA 平滑系数
        self.forget_score = np.zeros(self.num_classes, dtype=np.float64)

        self.use_drift_priority = bool(getattr(self.args, 'replay_drift', True))
        self.class_feat_init = [None for _ in range(self.num_classes)]
        self.class_feat_ema = [None for _ in range(self.num_classes)]
        self.class_drift_hist = [
            deque(maxlen=int(getattr(self.args, 'drift_hist_len', 10)))
            for _ in range(self.num_classes)
        ]
        self.class_drift_score = np.zeros(self.num_classes, dtype=np.float64)
        self.class_drift_ema_score = np.zeros(self.num_classes, dtype=np.float64)
        self.drift_anchor_weight = float(getattr(self.args, 'drift_anchor_weight', 0.05))
        self.drift_alpha = float(getattr(self.args, 'drift_alpha', 0.1))
        self.old_classes = []
        self.replay_use_count = np.zeros(self.num_classes, dtype=np.int64)
        self.drift_ema_task_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.drift_ema_task_cnt = np.zeros(self.num_classes, dtype=np.int64)
        self.drift_anchor_task_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.drift_anchor_task_cnt = np.zeros(self.num_classes, dtype=np.int64)
        if hasattr(self.args, 'domain_num'):
            self.history_cd_mask = torch.zeros(self.args.class_num, self.args.domain_num, dtype=torch.bool)
        else:
            self.history_cd_mask = torch.zeros(self.num_classes, 20, dtype=torch.bool)




    def _append_json_record(self, record, filename="drift_replay_log.jsonl"):
        if not (hasattr(self, "args") and getattr(self.args, "output_dir", None)):
            return
        if not utils.is_main_process():
            return

        log_path = os.path.join(self.args.output_dir, filename)
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[WARN] failed to write {filename}: {e}", file=sys.stderr)

    @torch.no_grad()
    def update_epoch_drift(self, model, device, task_id: int, epoch: int):
        if not (self.use_replay and self.use_drift_priority):
            return
        if task_id <= 0:
            return
        if len(getattr(self, "old_classes", [])) == 0:
            return
        if len(getattr(self, "replay_buffer", {})) == 0:
            return

        old_classes = list(self.old_classes)
        if len(old_classes) == 0:
            return

        max_per_class = int(getattr(self.args, 'drift_max_per_class', 64))
        drift_batch_size = int(getattr(self.args, 'drift_batch_size',
                                       getattr(self.args, 'batch_size', 4)))
        if drift_batch_size <= 0:
            drift_batch_size = 1

        was_training = model.training
        model.eval()

        any_sample = False

        for c in old_classes:
            c_int = int(c)
            buf = self.replay_buffer.get(c_int, [])
            # print(f"[DRIFT DEBUG] T{task_id} E{epoch} Class {c_int} buffer size: {len(buf)}")
            if len(buf) == 0:
                continue

            m = min(len(buf), max_per_class)
            if m <= 0:
                continue

            idxs = self.rng.choice(len(buf), size=m, replace=False)

            feat_sum_c = None
            cnt_c = 0

            for start_idx in range(0, m, drift_batch_size):
                end_idx = min(m, start_idx + drift_batch_size)
                batch_idxs = idxs[start_idx:end_idx]

                x_list = []
                for j in batch_idxs:
                    x_cpu, _ = buf[int(j)]
                    x_list.append(x_cpu)

                if len(x_list) == 0:
                    continue

                x_batch = torch.stack(x_list, dim=0).to(device, non_blocking=True)
                feats = model.forward_features(x_batch)[:, 0].detach().cpu()  # (B_chunk, D)

                if feat_sum_c is None:
                    feat_sum_c = feats.sum(dim=0)
                else:
                    feat_sum_c += feats.sum(dim=0)

                cnt_c += feats.shape[0]
                del x_batch, feats
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if feat_sum_c is None or cnt_c == 0:
                continue

            any_sample = True
            centroid = feat_sum_c / float(cnt_c)

            d_ema = 0.0
            d_anchor = 0.0

            prev = self.class_feat_ema[c_int]
            if prev is None:
                self.class_feat_ema[c_int] = centroid.clone()
                if self.class_feat_init[c_int] is None:
                    self.class_feat_init[c_int] = centroid.clone()
                drift = 0.0
            else:
                anchor = self.class_feat_init[c_int] if self.class_feat_init[c_int] is not None else prev
                diff_ema = centroid - prev
                diff_anchor = centroid - anchor
                raw_diff_ema_norm = torch.norm(diff_ema, p=2).item()
                raw_diff_anchor_norm = torch.norm(diff_anchor, p=2).item()

                # debug：
                # print(f"[DRIFT DEBUG] T{task_id} E{epoch} Cls{c_int}: "
                #       f"Raw d_ema={raw_diff_ema_norm:.10f}, "
                #       f"Raw d_anchor={raw_diff_anchor_norm:.10f}, "
                #       f"Centroid Norm={torch.norm(centroid, p=2).item():.10f}")

                d_ema = raw_diff_ema_norm
                d_anchor = raw_diff_anchor_norm
                lam = float(self.drift_anchor_weight)
                drift = (1.0 - lam) * d_ema + lam * d_anchor

                # update EMA center of mass
                alpha = float(self.drift_alpha)
                self.class_feat_ema[c_int] = alpha * prev + (1.0 - alpha) * centroid

            # Update the historical drift-goal.
            self.class_drift_hist[c_int].append(float(drift))
            if len(self.class_drift_hist[c_int]) > 0:
                self.class_drift_score[c_int] = float(np.mean(self.class_drift_hist[c_int]))
            else:
                self.class_drift_score[c_int] = 0.0

            self.class_drift_ema_score[c_int] = float(d_ema)
            self.drift_ema_task_sum[c_int] += float(d_ema)
            self.drift_ema_task_cnt[c_int] += 1
            self.drift_anchor_task_sum[c_int] += float(d_anchor)
            self.drift_anchor_task_cnt[c_int] += 1

            record = {
                "time": datetime.datetime.now().isoformat(),
                "type": "drift_epoch",
                "task": int(task_id),
                "epoch": int(epoch),
                "cls": int(c_int),
                "d_ema": float(d_ema),
                "d_anchor": float(d_anchor),
            }
            self._append_json_record(record)

        if was_training:
            model.train()

        # if not any_sample:
        #     print(f"[DRIFT DEBUG] T{task_id} E{epoch} WARNING: no samples used for drift.")

    @torch.no_grad()
    def _init_drift_centroids_for_old_classes(self, model, device, task_id, class_mask):
        """
        At the start of the current task:
        1) Construct a class-balanced sub-pool for all old classes from the current replay_buffer, with total capacity ≈ replay_size;
        2) Estimate the initial centroid μ_init on this sub-pool, initialize the EMA centroid μ_ema and drift history;
        3) Subsequently, update drift every few batches using samples of old classes from the batch, write to `class_drift_score[c]`,
            and use it as a priority signal in `_replay_sample`.
        """
        if not self.use_replay or not self.use_forget_priority or not self.use_drift_priority:
            return
        if task_id <= 0:
            return
        if class_mask is None:
            return

        old_classes = set()
        for t in range(task_id):
            for cid in class_mask[t]:
                old_classes.add(int(cid))
        if len(old_classes) == 0:
            self.old_classes = []
            return
        old_classes = sorted(list(old_classes))
        if self.use_replay and self.replay_size > 0:
            total_cap = int(self.replay_size)
            num_cls = len(old_classes)
            per_cls_base = max(total_cap // max(num_cls, 1), 1)
            new_buffer = {}
            used_counts = {}
            for c in old_classes:
                buf = self.replay_buffer.get(c, [])
                if len(buf) == 0:
                    continue
                m = min(len(buf), per_cls_base)
                idxs = self.rng.choice(len(buf), size=m, replace=False)
                new_buffer[c] = [buf[j] for j in idxs]
                used_counts[c] = m

            cur_total = sum(len(v) for v in new_buffer.values())
            remain = max(0, total_cap - cur_total)
            if remain > 0:
                pool = []
                for c in old_classes:
                    buf = self.replay_buffer.get(c, [])
                    used = used_counts.get(c, 0)
                    for j in range(used, len(buf)):
                        pool.append((c, buf[j]))
                if len(pool) > 0:
                    idxs = self.rng.choice(len(pool), size=min(remain, len(pool)), replace=False)
                    for k in idxs:
                        c, sample = pool[k]
                        new_buffer.setdefault(c, []).append(sample)

            self.replay_buffer = new_buffer

        for c in old_classes:
            self.class_feat_init[c] = None
            self.class_feat_ema[c] = None
            self.class_drift_hist[c].clear()
            self.class_drift_score[c] = 0.0
            self.class_drift_ema_score[c] = 0.0

        max_per_class = int(getattr(self.args, 'drift_max_per_class', 64))
        drift_batch_size = int(getattr(self.args, 'drift_batch_size', getattr(self.args, 'batch_size', 4)))

        was_training = model.training
        model.eval()

        any_sample = False

        with torch.no_grad():
            for c in old_classes:
                buf = self.replay_buffer.get(c, [])
                if len(buf) == 0:
                    continue
                m = min(len(buf), max_per_class)
                if m <= 0:
                    continue
                idxs = self.rng.choice(len(buf), size=m, replace=False)
                feat_sum_c = None
                cnt_c = 0

                for start_idx in range(0, m, drift_batch_size):
                    end_idx = min(m, start_idx + drift_batch_size)
                    batch_idxs = idxs[start_idx:end_idx]

                    x_list = []
                    for j in batch_idxs:
                        x_cpu, _ = buf[j]
                        x_list.append(x_cpu)
                    if len(x_list) == 0:
                        continue

                    x_batch = torch.stack(x_list, dim=0).to(device, non_blocking=True)
                    feats = model.forward_features(x_batch)[:, 0].detach().cpu()  # (B_chunk, D)
                    if feat_sum_c is None:
                        feat_sum_c = feats.sum(dim=0)
                    else:
                        feat_sum_c += feats.sum(dim=0)
                    cnt_c += feats.shape[0]
                if feat_sum_c is None or cnt_c == 0:
                    continue
                any_sample = True
                centroid = feat_sum_c / float(cnt_c)

                self.class_feat_init[c] = centroid.clone()
                self.class_feat_ema[c] = centroid.clone()
                print(
                    f"[DRIFT DEBUG INIT] Class {c} initialized. Centroid Norm: {torch.norm(centroid, p=2).item():.6f}")

        if was_training:
            model.train()
        self.old_classes = list(old_classes)
        if not any_sample:
            print("[DRIFT WARN] no old-class samples for initial centroid; drift disabled for this task.")
            self.old_classes = old_classes
            return

    @torch.no_grad()
    def _class_counts_from_loader(self, data_loader):
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                _, targets, _ = batch
            else:
                _, targets = batch
            t = targets.view(-1).to('cpu')
            counts.index_add_(0, t, torch.ones_like(t, dtype=torch.long))
        return counts

    @torch.no_grad()
    def _pair_counts_from_loader(self, data_loader):
        counts = torch.zeros(self.num_classes, self.args.domain_num, dtype=torch.long)
        for batch in data_loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) == 3):
                continue
            _, targets, domains = batch
            t = targets.view(-1).to('cpu')
            d = domains.view(-1).to('cpu')
            for ci, di in zip(t, d):
                counts[int(ci), int(di)] += 1
        return counts

    @torch.no_grad()
    def _compute_rebalance_weights(self, train_loader):
        """
        Calculate and cache weights based on args.rb_level:
        - class: self.loss_weight(C,)
        - domain: self.domain_loss_weight(D,)
        - class_domain: self.pair_loss_weight (C,D)
        Perform mean normalization only on “occurred positions” to average=1, with optional upper bound clipping.
        """
        mode = getattr(self.args, 'rb_mode', 'effective_num')
        beta = float(getattr(self.args, 'rb_beta', 0.9999))
        cap = float(getattr(self.args, 'rb_cap', 10.0))
        level = getattr(self.args, 'rb_level', 'class')

        def _w_from_counts(cnt):
            present = cnt > 0
            w = torch.zeros_like(cnt, dtype=torch.float)
            if mode == 'inverse_freq':
                w[present] = 1.0 / cnt[present].float()
            else:
                eff = 1.0 - torch.pow(beta, cnt[present].float())
                w[present] = (1.0 - beta) / eff
            if present.any():
                w[present] = w[present] * (present.sum() / w[present].sum())
            if cap > 0:
                w[present] = torch.clamp(w[present], max=cap)
            return w

        if level == 'class':
            counts = self._class_counts_from_loader(train_loader)
            self.loss_weight = _w_from_counts(counts).to(self.device)
            self.pair_loss_weight = None
            self.domain_loss_weight = None
        elif level == 'domain':
            pair = self._pair_counts_from_loader(train_loader)
            counts_d = pair.sum(dim=0)  # (D,)
            self.domain_loss_weight = _w_from_counts(counts_d).to(self.device)
            self.loss_weight = None
            self.pair_loss_weight = None
        else:  # 'class_domain'
            pair = self._pair_counts_from_loader(train_loader).float()
            present = pair > 0
            W = torch.zeros_like(pair, dtype=torch.float)
            if present.any():
                if mode == 'inverse_freq':
                    W[present] = 1.0 / pair[present]
                else:
                    eff = 1.0 - torch.pow(beta, pair[present])
                    W[present] = (1.0 - beta) / eff
                W[present] = W[present] * (present.sum() / W[present].sum())
                if cap > 0:
                    W[present] = torch.clamp(W[present], max=cap)
            self.pair_loss_weight = W.to(self.device)
            self.loss_weight = None
            self.domain_loss_weight = None

    def _replay_enqueue(self, x: torch.Tensor, y: torch.Tensor):
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        for i in range(y_cpu.shape[0]):
            c = int(y_cpu[i].item())
            self.class_seen[c] += 1
            self.replay_buffer.setdefault(c, [])
            self.replay_buffer[c].append((x_cpu[i], y_cpu[i]))

        def _total_size():
            return sum(len(v) for v in self.replay_buffer.values())

        while _total_size() > self.replay_size:
            max_c = max(self.replay_buffer.keys(), key=lambda k: len(self.replay_buffer[k]))
            idx = self.rng.randint(len(self.replay_buffer[max_c]))
            self.replay_buffer[max_c].pop(idx)
            if len(self.replay_buffer[max_c]) == 0:
                del self.replay_buffer[max_c]

    @torch.no_grad()
    def _update_forget_from_replay(self, model, device: torch.device):
        if not self.use_forget_priority:
            return
        if not self.use_replay:
            return
        if len(self.replay_buffer) == 0:
            return

        xs, ys = [], []
        for c, buf in self.replay_buffer.items():
            if len(buf) == 0:
                continue
            for j in range(len(buf)):
                x_cpu, y_cpu = buf[j]
                xs.append(x_cpu)
                ys.append(int(y_cpu.item()))

        if len(xs) == 0:
            return

        x = torch.stack(xs, dim=0).to(device, non_blocking=True)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        model.eval()
        logits = model(x)
        logits, _, _ = self.get_max_label_logits(logits,class_mask=None,task_id=None,slice=True)
        ce_vec = F.cross_entropy(logits, y, reduction='none').detach().cpu().numpy()

        sum_loss = np.zeros(self.num_classes, dtype=np.float64)
        cnt = np.zeros(self.num_classes, dtype=np.float64)
        for loss, c in zip(ce_vec, ys):
            sum_loss[c] += float(loss)
            cnt[c] += 1.0

        alpha = float(self.forget_alpha)
        for c in range(self.num_classes):
            if cnt[c] == 0:
                continue
            avg_loss = sum_loss[c] / cnt[c]
            old = self.forget_score[c]
            if old == 0.0:
                self.forget_score[c] = avg_loss
            else:
                self.forget_score[c] = (1.0 - alpha) * old + alpha * avg_loss

    def _replay_sample(self, n: int, device: torch.device):

        if n <= 0:
            return None

        avail_classes = [c for c, buf in self.replay_buffer.items() if len(buf) > 0]

        if hasattr(self, "added_classes_in_cur_task"):
            cur_new = self.added_classes_in_cur_task
            avail_classes = [c for c in avail_classes if c not in cur_new]

        if len(avail_classes) == 0:
            return None

        cls_arr = np.array(avail_classes, dtype=np.int64)

        priority_p = None
        if self.use_forget_priority and getattr(self, 'forget_weight', 0.0) > 0:
            if getattr(self, 'use_drift_priority', False):
                raw = np.array(
                    [max(float(self.class_drift_score[int(c)]), 0.0) for c in avail_classes],
                    dtype=np.float64
                )
            else:
                raw = np.array(
                    [max(float(self.forget_score[int(c)]), 0.0) for c in avail_classes],
                    dtype=np.float64
                )

            if raw.sum() > 0:
                priority_p = raw / raw.sum()
            else:
                priority_p = None
        n_forget = 0
        n_uniform = n
        if priority_p is not None:
            w_f = float(self.forget_weight)
            w_f = max(0.0, min(1.0, w_f))
            n_forget = int(round(n * w_f))
            n_uniform = n - n_forget

        xs, ys = [], []
        for _ in range(n_uniform):
            c = int(self.rng.choice(cls_arr))
            buf = self.replay_buffer[c]
            j = self.rng.randint(len(buf))
            x_cpu, y_cpu = buf[j]
            xs.append(x_cpu)
            ys.append(y_cpu)
        if n_forget > 0 and priority_p is not None:
            for _ in range(n_forget):
                c = int(self.rng.choice(cls_arr, p=priority_p))
                buf = self.replay_buffer[c]
                j = self.rng.randint(len(buf))
                x_cpu, y_cpu = buf[j]
                xs.append(x_cpu)
                ys.append(y_cpu)

        if len(xs) == 0:
            return None

        x = torch.stack(xs, dim=0).to(device, non_blocking=True)
        y = torch.stack(ys, dim=0).to(device, non_blocking=True)
        return x, y

    def kl_div(self, p, q):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        kl = torch.mean(torch.sum(p * torch.log(p / q), dim=1))
        return kl

    def set_new_head(self, model, labels_to_be_added, task_id):
        len_new_nodes = len(labels_to_be_added)
        self.labels_in_head = np.concatenate((self.labels_in_head, labels_to_be_added))
        self.added_classes_in_cur_task.update(labels_to_be_added)
        self.head_timestamps = np.concatenate((self.head_timestamps, [task_id] * len_new_nodes))
        prev_weight, prev_bias = model.head.weight, model.head.bias
        prev_shape = prev_weight.shape  # (class, dim)
        new_head = torch.nn.Linear(prev_shape[-1], prev_shape[0] + len_new_nodes)

        new_head.weight[:prev_weight.shape[0]].data.copy_(prev_weight)
        new_head.weight[prev_weight.shape[0]:].data.copy_(prev_weight[labels_to_be_added])
        new_head.bias[:prev_weight.shape[0]].data.copy_(prev_bias)
        new_head.bias[prev_weight.shape[0]:].data.copy_(prev_bias[labels_to_be_added])

        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added})")
        return new_head

    def inference_acc(self, model, data_loader, device):
        print("Start detecting labels to be added...")
        accuracy_per_label = []
        correct_pred_per_label = [0 for _ in range(len(self.current_classes))]
        num_instance_per_label = [0 for _ in range(len(self.current_classes))]

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if self.args.develop and batch_idx > 200:
                    break

                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(input)

                if output.shape[-1] > self.num_classes:
                    output, _, _ = self.get_max_label_logits(output,self.current_classes,task_id=self.current_task,slice=True)

                mask = self.current_classes
                not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64, device=device)
                logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
                _, pred = torch.max(logits, 1)

                correct_predictions = (pred == target)
                for i, label in enumerate(self.current_classes):
                    cls_mask = (target == label)
                    num_correct_pred = torch.sum(correct_predictions[cls_mask])
                    correct_pred_per_label[i] += num_correct_pred.item()
                    num_instance_per_label[i] += cls_mask.sum().item()

        for correct, num in zip(correct_pred_per_label, num_instance_per_label):
            if num > 0:
                accuracy_per_label.append(round(correct / num, 2))
            else:
                accuracy_per_label.append(0.0)
        return accuracy_per_label

    def _update_and_get_block_thresholds(self, data_loader):

        current_pair_counts = self._pair_counts_from_loader(data_loader)
        current_block_mask = (current_pair_counts > 0)
        num_cur_blocks = current_block_mask.sum().item()

        if self.history_cd_mask.device != current_block_mask.device:
            self.history_cd_mask = self.history_cd_mask.to(current_block_mask.device)
        self.history_cd_mask = self.history_cd_mask | current_block_mask
        num_total_blocks = self.history_cd_mask.sum().item()

        dyn_task_min = 1.0 / num_cur_blocks if num_cur_blocks > 0 else 0.0
        dyn_total_min = 1.0 / num_total_blocks if num_total_blocks > 0 else 0.0

        print(f"[Dynamic IC Thresholds] Current Blocks: {num_cur_blocks} (Thre: {dyn_task_min:.5f}) | "
              f"Total Blocks: {num_total_blocks} (Thre: {dyn_total_min:.5f})")

        return dyn_task_min, dyn_total_min

    def detect_labels_to_be_added(self, inference_acc, thresholds=[],dyn_task_min=None, dyn_total_min=None):
        labels_with_low_accuracy = []
        # min_task = float(getattr(self.args, 'ic_tail_task_min', 0.0))
        # min_total = float(getattr(self.args, 'ic_tail_total_min', 0.0))
        if getattr(self.args, 'classwise_tau', True):
            min_task = dyn_task_min
            min_total = dyn_total_min
            # print(f"[Evidence Gating] ON. Thresholds: task={min_task:.4f}, total={min_total:.4f}")
        else:
            min_task = 0.0
            min_total = 0.0
        total_task_samples = self.cur_class_counts.sum()
        total_all_samples = self.cum_counts_total.sum()

        if self.args.d_threshold:
            triplets = zip(self.current_classes, inference_acc, thresholds)
        else:
            triplets = ((l, a, self.args.thre) for l, a in zip(self.current_classes, inference_acc))

        for label, acc, thre in triplets:
            cond_acc = (acc <= thre)
            n_task = int(self.cur_class_counts[label])
            n_total = int(self.cum_counts_total[label])

            task_ratio = n_task / total_task_samples if total_task_samples > 0 else 0
            total_ratio = n_total / total_all_samples if total_all_samples > 0 else 0
            if task_ratio < min_task or total_ratio < min_total:
                continue
            if cond_acc:
                labels_with_low_accuracy.append(label)

        print(f"Labels whose node to be increased (after evidence gating): {labels_with_low_accuracy}")
        return labels_with_low_accuracy

    def find_same_cluster_items(self, vec):
        v = vec.detach().cpu()

        mask = torch.isnan(v) | torch.isinf(v)
        if mask.any():
            finite = v[~mask]
            if finite.numel() == 0:
                v = torch.zeros_like(v)
                print("[CAST] all values in diff_adapter are NaN/Inf, fill with 0.")
            else:
                fill_val = finite.mean()
                v[mask] = fill_val
                print(f"[CAST] Impute NaN/Inf in diff_adapter with mean = {fill_val.item():.4e}")

        v_np = v.numpy().astype(float).reshape(1, -1)

        if self.kmeans.n_clusters == 1:
            other_cluster_vecs = self.adapter_vec_array
            other_cluster_vecs = torch.tensor(other_cluster_vecs, dtype=torch.float32).to(self.device)
            same_cluster_vecs = None
        else:
            predicted_cluster = self.kmeans.predict(vec.unsqueeze(0).detach().cpu())[0]
            same_cluster_vecs = self.adapter_vec_array[self.cluster_assignments == predicted_cluster]
            other_cluster_vecs = self.adapter_vec_array[self.cluster_assignments != predicted_cluster]
            same_cluster_vecs = torch.tensor(same_cluster_vecs, dtype=torch.float32).to(self.device)
            other_cluster_vecs = torch.tensor(other_cluster_vecs, dtype=torch.float32).to(self.device)
        return same_cluster_vecs, other_cluster_vecs

    def calculate_l2_distance(self, diff_adapter, other):
        weights = []
        for o in other:
            l2_distance = torch.norm(diff_adapter - o, p=2)
            weights.append(l2_distance.item())
        weights = torch.tensor(weights)
        weights = weights / torch.sum(weights)  # summation-> 1
        return weights

    def train_one_epoch(self, model: torch.nn.Module,
                        criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0,
                        set_training_mode=True, task_id=-1, class_mask=None, ema_model=None, args=None, ):

        model.train(set_training_mode)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

        for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop and batch_idx > 20:
                break

            # 兼容 (x,y) 与 (x,y,domain)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                input, target, domain = batch
                domain = domain.to(device, non_blocking=True).long()
            else:
                input, target = batch
                domain = None

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if self.use_replay:
                self._replay_enqueue(input, target)
            if self.use_replay:
                batch_size = int(input.shape[0])
                is_last_epoch = ((epoch + 1) == args.epochs)
                if is_last_epoch:
                    replay_k = int(round(batch_size * 0.5))
                    replay_k = max(0, min(replay_k, batch_size))
                    keep_k = batch_size - replay_k
                else:
                    replay_k = int(round(batch_size * self.replay_ratio))
                    replay_k = max(0, min(replay_k, batch_size))
                    keep_k = batch_size - replay_k

                replay_pair = self._replay_sample(replay_k, device)
                if replay_pair is not None:
                    rx, ry = replay_pair
                    if task_id >= 1 and hasattr(self, "replay_use_count"):
                        with torch.no_grad():
                            cls_np = ry.detach().cpu().numpy().astype(np.int64)
                        for c in cls_np:
                            if 0 <= c < self.replay_use_count.shape[0]:
                                self.replay_use_count[c] += 1
                    if keep_k == 0:
                        input, target = rx, ry
                        domain = None
                    else:
                        perm = torch.randperm(batch_size, device=device)
                        sel = perm[:keep_k]
                        input_cur = input[sel]
                        target_cur = target[sel]
                        input = torch.cat([input_cur, rx], dim=0)
                        target = torch.cat([target_cur, ry], dim=0)
                        domain = None

            output = model(input)  # (bs, class + n)
            distill_loss = 0
            if self.distill_head is not None:
                feature = model.forward_features(input)[:, 0]
                output_distill = self.distill_head(feature)
                mask_nodes = torch.isin(torch.tensor(self.labels_in_head), torch.tensor(self.current_classes))
                cur_class_nodes = torch.where(mask_nodes)[0]
                m_added = torch.isin(torch.tensor(self.labels_in_head[cur_class_nodes]),
                                     torch.tensor(list(self.added_classes_in_cur_task)))
                distill_node_indices = self.labels_in_head[cur_class_nodes][~m_added]
                distill_loss = self.kl_div(output[:, distill_node_indices], output_distill[:, distill_node_indices])

            if output.shape[-1] > self.num_classes:
                output, _, _ = self.get_max_label_logits(output, class_mask[task_id], slice=False)
                if len(self.added_classes_in_cur_task) > 0:
                    for added_class in self.added_classes_in_cur_task:
                        cur_node = np.where(self.labels_in_head == added_class)[0][-1]
                        output[:, added_class] = output[:, cur_node]
                output = output[:, :self.num_classes]

            if args.train_mask and class_mask is not None:
                base_mask = set(int(x) for x in class_mask[task_id])
                tgt_labels = set(int(x) for x in torch.unique(target).detach().cpu().tolist())
                eff_mask = sorted(base_mask | tgt_labels)
                all_ids = np.arange(self.num_classes)
                not_mask = np.setdiff1d(all_ids, np.array(eff_mask, dtype=np.int64))
                not_mask = torch.tensor(not_mask, dtype=torch.int64, device=device)
                logits = output
                if not_mask.numel() > 0:
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            else:
                logits = output

            rb_mode = getattr(args, 'rb_mode', 'effective_num')
            use_la_train = bool(getattr(args, 'use_la_train', False))


            logits_for_ce = logits

            if getattr(args, 'rebalance', False) and rb_mode == 'balanced_softmax':
                counts = torch.tensor(np.maximum(self.cum_counts_total, 1),
                                      dtype=torch.float32, device=device)
                logits_bs = logits_for_ce + torch.log(counts).unsqueeze(0)
                ce_loss = F.cross_entropy(logits_bs, target)

            elif use_la_train:
                counts = torch.tensor(np.maximum(self.cum_counts_total, 1),
                                      dtype=torch.float32, device=device)
                pi = counts / counts.sum()
                la_tau = float(getattr(args, 'la_tau', 1.0))
                logits_la = logits_for_ce - la_tau * torch.log(pi).unsqueeze(0)
                ce_loss = F.cross_entropy(logits_la, target)

            elif getattr(args, 'rebalance', False):
                level = getattr(args, 'rb_level', 'class')
                if level == 'class' and (getattr(self, 'loss_weight', None) is not None):
                    ce_loss = F.cross_entropy(logits_for_ce, target,
                                              weight=self.loss_weight)
                else:
                    ce_vec = F.cross_entropy(logits_for_ce, target, reduction='none')
                    if (level == 'class_domain'
                            and (getattr(self, 'pair_loss_weight', None) is not None)
                            and (domain is not None)):
                        w = self.pair_loss_weight[target, domain]  # (bs,)
                        ce_loss = (w * ce_vec).mean()
                    elif (level == 'domain'
                          and (getattr(self, 'domain_loss_weight', None) is not None)
                          and (domain is not None)):
                        w = self.domain_loss_weight[domain]  # (bs,)
                        ce_loss = (w * ce_vec).mean()
                    else:
                        ce_loss = ce_vec.mean()
            else:
                ce_loss = F.cross_entropy(logits_for_ce, target)
            loss = ce_loss

            if self.args.use_cast_loss:
                if len(self.adapter_vec) > args.k:
                    cur_adapters = model.get_adapter()
                    self.cur_adapters = self.flatten_parameters(cur_adapters)
                    diff_adapter = self.cur_adapters - self.prev_adapters

                    if torch.isnan(diff_adapter).any() or torch.isinf(diff_adapter).any():
                        print(f"[Warning] NaN/Inf detected in adapter weights at step {batch_idx}. Skipping CAST loss.")
                        # sys.exit(1) # 可选：如果参数已经坏了，可能需要停止训练
                    else:
                        _, other = self.find_same_cluster_items(diff_adapter)
                        sim = 0
                        weights = self.calculate_l2_distance(diff_adapter, other)
                        for o, w in zip(other, weights):
                            if self.args.norm_cast:
                                sim += w * torch.matmul(diff_adapter, o) / (torch.norm(diff_adapter) * torch.norm(o))
                            else:
                                sim += w * torch.matmul(diff_adapter, o)
                        orth_loss = args.beta * torch.abs(sim)
                        if self.args.use_cast_loss and orth_loss > 0:
                            loss += orth_loss

            if self.args.IC and distill_loss > 0:
                loss += distill_loss

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            if ema_model is not None:
                ema_model.update(model.get_adapter())

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def get_max_label_logits(self, output, class_mask, task_id=None, slice=False, target=None):
        mode = getattr(self.args, 'pooler', 'max')
        tau_base = float(getattr(self.args, 'pool_tau', 1.0))
        use_classwise = bool(getattr(self.args, 'classwise_tau', False))
        tau_gamma = float(getattr(self.args, 'tau_gamma', 0.5))

        if use_classwise:
            counts = torch.tensor(np.maximum(self.cum_counts_total, 1), dtype=torch.float32, device=output.device)
            norm = counts / counts.mean().clamp(min=1e-6)
            tau_vec = tau_base * torch.pow(norm.clamp(min=1e-3), -tau_gamma)

        def _agg(x, idx, label):
            if len(idx) == 1:
                return x[:, idx[0]]
            if mode == 'mean':
                return x[:, idx].mean(dim=1)
            if mode == 'lse':
                t = (tau_vec[label] if use_classwise else tau_base)
                t = float(t.item()) if torch.is_tensor(t) else float(t)
                return torch.logsumexp(x[:, idx] / t, dim=1) * t
            return x[:, idx].max(dim=1)[0]

        for label in range(self.num_classes):
            label_nodes = np.where(self.labels_in_head == label)[0]
            if len(label_nodes) == 0:
                continue
            output[:, label] = _agg(output, label_nodes, label)

        if slice:
            output = output[:, :self.num_classes]
        return output, 0, 0

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader,
                 device, task_id=-1, class_mask=None, ema_model=None, args=None, ):
        per_class_correct = np.zeros(self.num_classes, dtype=np.int64)
        per_class_total = np.zeros(self.num_classes, dtype=np.int64)

        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id + 1)

        model.eval()

        correct_sum, total_sum = 0, 0
        label_correct, label_total = np.zeros((self.class_group_size)), np.zeros((self.class_group_size))
        per_class_correct = np.zeros(self.num_classes, dtype=np.int64)
        per_class_total = np.zeros(self.num_classes, dtype=np.int64)


        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop and batch_idx > 20:
                    break
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                logits_main = model(input)
                logits_main, _, _ = self.get_max_label_logits(logits_main, class_mask[task_id], task_id=task_id, target=target, slice=True )

                not_mask_tensor = None
                if args.train_mask and class_mask is not None:
                    base_mask = set(int(x) for x in class_mask[task_id])
                    tgt_labels = set(int(x) for x in torch.unique(target).detach().cpu().tolist())
                    eff_mask = sorted(base_mask | tgt_labels)
                    not_mask_np = np.setdiff1d(np.arange(self.num_classes), np.array(eff_mask, dtype=np.int64))
                    not_mask_tensor = torch.tensor(not_mask_np, dtype=torch.int64, device=device)
                    if not_mask_tensor.numel() > 0:
                        logits_main = logits_main.index_fill(dim=1, index=not_mask_tensor, value=float('-inf'))

                if bool(getattr(self.args, 'use_la_eval', False)):
                    counts_eval = torch.tensor(np.maximum(self.cum_counts_total, 1),
                                               dtype=torch.float32, device=device)
                    pi_eval = counts_eval / counts_eval.sum()
                    la_tau = float(getattr(self.args, 'la_tau', 1.0))
                    logits_main = logits_main - la_tau * torch.log(pi_eval).unsqueeze(0)
                outputs_for_ensemble = [logits_main.softmax(dim=1)]

                if ema_model is not None:
                    tmp_adapter = model.get_adapter()
                    model.put_adapter(ema_model.module)

                    logits_ema = model(input)
                    logits_ema, _, _ = self.get_max_label_logits(
                        logits_ema, class_mask[task_id], slice=True
                    )
                    if args.train_mask and class_mask is not None and not_mask_tensor is not None and not_mask_tensor.numel() > 0:
                        logits_ema = logits_ema.index_fill(dim=1, index=not_mask_tensor, value=float('-inf'))

                    outputs_for_ensemble.append(logits_ema.softmax(dim=1))
                    model.put_adapter(tmp_adapter)

                final_output = torch.stack(outputs_for_ensemble, dim=-1).max(dim=-1)[0]
                loss = criterion(final_output, target)
                if self.args.d_threshold and self.current_task + 1 != self.args.num_tasks and self.current_task == task_id:
                    label_correct, label_total = self.update_acc_per_label(label_correct, label_total, final_output,target)
                acc1, acc5 = accuracy(final_output, target, topk=(1, 5))
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
                try:
                    preds = final_output.argmax(dim=1)
                    y_np = target.detach().cpu().numpy().astype(np.int64)
                    p_np = preds.detach().cpu().numpy().astype(np.int64)

                    binc_total = np.bincount(y_np, minlength=self.num_classes)
                    binc_correct = np.bincount(
                        y_np,
                        weights=(p_np == y_np).astype(np.int64),
                        minlength=self.num_classes,)
                    per_class_total += binc_total
                    per_class_correct += binc_correct.astype(np.int64)
                except Exception:
                    pass

            if total_sum > 0:
                print(f"Max Pooling acc: {correct_sum / total_sum}")
            if self.args.d_threshold and task_id == self.current_task:
                domain_idx = int(self.label_train_count[self.current_classes][0])
                self.acc_per_label[self.current_classes, domain_idx] += np.round(label_correct / label_total,decimals=3)
                print(self.label_train_count)
                print(self.acc_per_label)
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                      losses=metric_logger.meters['Loss']))
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if per_class_total.sum() > 0:
            per_class_acc = (
                    per_class_correct.astype(np.float64)
                    / np.maximum(per_class_total, 1)
            )
            stats["_per_class_total"] = per_class_total
            stats["_per_class_correct"] = per_class_correct
            stats["_per_class_acc"] = per_class_acc

        return stats

    @torch.no_grad()
    def evaluate_till_now(self, model: torch.nn.Module, data_loader,
                          device, task_id=-1, class_mask=None, acc_matrix=None, ema_model=None, args=None, ):
        stat_matrix = np.zeros((3, args.num_tasks))
        global_per_class_correct = np.zeros(self.num_classes, dtype=np.int64)
        global_per_class_total = np.zeros(self.num_classes, dtype=np.int64)

        for i in range(task_id + 1):
            test_stats = self.evaluate(model=model, data_loader=data_loader[i]['val'],
                                       device=device, task_id=i, class_mask=class_mask, ema_model=ema_model, args=args)
            stat_matrix[0, i] = test_stats['Acc@1']
            stat_matrix[1, i] = test_stats['Acc@5']
            stat_matrix[2, i] = test_stats['Loss']
            acc_matrix[i, task_id] = test_stats['Acc@1']

            if "_per_class_acc" in test_stats and "_per_class_total" in test_stats:
                per_class_acc = test_stats["_per_class_acc"]
                per_class_total = test_stats["_per_class_total"]
                acc_dict = {
                    int(c): float(per_class_acc[c])
                    for c in range(len(per_class_total))
                    if per_class_total[c] > 0
                }
                record = {
                    "time": datetime.datetime.now().isoformat(),
                    "type": "eval_acc",
                    "task_train": int(task_id),
                    "task_eval": int(i),
                    "per_class_acc": acc_dict,
                }
                self._append_json_record(record)

        avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
        diagonal = np.diag(acc_matrix)
        result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
            task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2]
        )

        forgetting = 0.0
        backward = 0.0
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) -acc_matrix[:, task_id])[:task_id])
            backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
            result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)

        print(result_str)
        if global_per_class_total.sum() > 0:
            per_class_acc_global = np.divide(
                global_per_class_correct.astype(np.float64),
                np.maximum(global_per_class_total, 1),
            )
            self.last_eval_class_acc = per_class_acc_global
        else:
            self.last_eval_class_acc = None
        test_stats['Avg_Acc1'] = float(avg_stat[0])
        test_stats['Avg_Acc5'] = float(avg_stat[1])
        test_stats['Avg_Loss'] = float(avg_stat[2])
        test_stats['Forgetting'] = float(forgetting)
        test_stats['Backward'] = float(backward)
        test_stats.pop("_per_class_total", None)
        test_stats.pop("_per_class_correct", None)
        test_stats.pop("_per_class_acc", None)

        return test_stats

    def log_drift_and_replay_stats(self, task_id, args):
        if not (args.output_dir and utils.is_main_process()):
            return

        log_file = os.path.join(args.output_dir, "drift_replay_log.txt")
        cls_set = set()

        if hasattr(self, "replay_buffer"):
            for c, buf in self.replay_buffer.items():
                try:
                    if len(buf) > 0:
                        cls_set.add(int(c))
                except TypeError:
                    continue

        for c in getattr(self, "old_classes", []):
            try:
                cls_set.add(int(c))
            except Exception:
                continue

        cur_cls = getattr(self, "current_classes", None)
        if cur_cls is not None:
            try:
                for c in list(cur_cls):
                    cls_set.add(int(c))
            except TypeError:
                try:
                    cls_set.add(int(cur_cls))
                except Exception:
                    pass
        if len(cls_set) == 0:
            return

        cls_list = sorted(cls_set)
        drift_scores = {int(c): float(self.class_drift_score[int(c)]) for c in cls_list}
        ema_arr = getattr(self, "class_drift_ema_score", self.class_drift_score)
        ema_drift_scores = {int(c): float(ema_arr[int(c)]) for c in cls_list}
        drift_total = float(np.sum(self.class_drift_score))
        ema_total = float(np.sum(ema_arr))

        priority_scores = None
        if self.use_forget_priority and float(getattr(self, "forget_weight", 0.0)) > 0.0 and len(cls_list) > 0:
            if self.use_drift_priority:
                raw = np.array(
                    [max(float(self.class_drift_score[int(c)]), 0.0) for c in cls_list],
                    dtype=np.float64,
                )
            else:
                raw = np.array(
                    [max(float(self.forget_score[int(c)]), 0.0) for c in cls_list],
                    dtype=np.float64,
                )
            if raw.sum() > 0:
                prob = raw / raw.sum()
                priority_scores = {int(c): float(prob[i]) for i, c in enumerate(cls_list)}

        per_class_acc = {}
        acc_arr = getattr(self, "last_eval_class_acc", None)
        if acc_arr is not None:
            for c in cls_list:
                idx = int(c)
                if 0 <= idx < len(acc_arr):
                    per_class_acc[idx] = float(acc_arr[idx])

        record = {
            "time": datetime.datetime.now().isoformat(),
            "task": int(task_id),
            "drift_total": drift_total,
            "drift_ema_total": ema_total,
            "per_class_drift": drift_scores,
            "per_class_ema_drift": ema_drift_scores,
            "per_class_priority": priority_scores,
            "per_class_acc": per_class_acc,
        }

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[WARN] Failed to write drift_replay_log: {e}", file=sys.stderr)


    def flatten_parameters(self, modules):
        flattened_params = []

        for m in modules:
            params = list(m.parameters())
            flattened_params.extend(params)
        return torch.cat([param.view(-1) for param in flattened_params])

    def cluster_adapters(self):
        k = self.args.k
        if len(self.adapter_vec) > k:
            self.adapter_vec_array = torch.stack(self.adapter_vec).detach().cpu().numpy().astype(float)

            mask = ~np.isfinite(self.adapter_vec_array)
            if mask.any():

                col_mean = np.nanmean(
                    np.where(mask, np.nan, self.adapter_vec_array),
                    axis=0
                )
                rows, cols = np.where(mask)
                self.adapter_vec_array[rows, cols] = col_mean[cols]
                print("[CAST] Imputed NaN/Inf in adapter_vec_array with column means.")

            self.kmeans = KMeans(n_clusters=k, n_init=10)
            self.kmeans.fit(self.adapter_vec_array)
            self.cluster_assignments = self.kmeans.labels_
            print("Cluster(shifts) Assignments:", self.cluster_assignments)

    def pre_train_epoch(self, model: torch.nn.Module, epoch: int = 0, task_id: int = 0, args=None, ):
        if task_id == 0 or args.num_freeze_epochs < 1:
            return model

        if epoch == 0:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = False
            print('Freezing adapter parameters for {} epochs'.format(args.num_freeze_epochs))

        if epoch == args.num_freeze_epochs:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = True
            print('Unfreezing adapter parameters')
        return model

    def pre_train_task(self, model, data_loader, device, task_id, args):
        self.current_task += 1
        self.current_class_group = int(min(self.class_mask[task_id]) / self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]
        with torch.no_grad():
            counts = self._class_counts_from_loader(data_loader)
        self.cur_class_counts = counts.detach().cpu().numpy().astype(np.int64)
        if getattr(self, 'cum_counts_total', None) is None:
            self.cum_counts_total = np.zeros(self.num_classes, dtype=np.int64)
        self.cum_counts_total[:len(self.cur_class_counts)] += self.cur_class_counts

        print(f"\n\nTASK : {task_id}")
        self.added_classes_in_cur_task = set()
        # ! distillation
        if self.class_group_train_count[self.current_class_group] == 0:
            self.distill_head = None
        else:  # already seen classes
            if self.args.IC:
                self.distill_head = self.classifier_pool[self.current_class_group]
                inf_acc = self.inference_acc(model, data_loader, device)
                dyn_task_min, dyn_total_min = self._update_and_get_block_thresholds(data_loader)
                thresholds = []
                if self.args.d_threshold:
                    count = self.class_group_train_count[self.current_class_group]
                    if count > 0:
                        average_accs = np.sum(self.acc_per_label[self.current_classes, :count], axis=1) / count
                    thresholds = self.args.gamma * (average_accs - inf_acc) / average_accs
                    thresholds = self.tanh(torch.tensor(thresholds)).tolist()
                    thresholds = [round(t, 2) if t > self.args.thre else self.args.thre for t in thresholds]
                    print(f"Thresholds for class {self.current_classes[0]}~{self.current_classes[-1]} : {thresholds}")
                labels_to_be_added = self.detect_labels_to_be_added(
                    inf_acc,
                    thresholds,
                    dyn_task_min=dyn_task_min,
                    dyn_total_min=dyn_total_min
                )

                if len(labels_to_be_added) > 0:  # ! Add node to the classifier if needed
                    new_head = self.set_new_head(model, labels_to_be_added, task_id).to(device)
                    model.head = new_head
        optimizer = create_optimizer(args, model)

        with torch.no_grad():
            prev_adapters = model.get_adapter()
            self.prev_adapters = self.flatten_parameters(prev_adapters)
            self.prev_adapters.requires_grad = False

        if task_id == 0:
            self.task_type_list.append("Initial")
            if getattr(args, 'rebalance', False):
                self._compute_rebalance_weights(data_loader)
            return model, optimizer

        prev_class = self.class_mask[task_id - 1]
        cur_class = self.class_mask[task_id]
        self.cur_domain = self.domain_list[task_id]
        self.task_type = "DIL" if (prev_class == cur_class) else "CIL"
        self.task_type_list.append(self.task_type)
        print(f"Current task : {self.task_type}")
        if getattr(args, 'rebalance', False):
            self._compute_rebalance_weights(data_loader)

        return model, optimizer

    def post_train_task(self, model: torch.nn.Module, task_id=-1):
        self.class_group_train_count[self.current_class_group] += 1
        self.classifier_pool[self.current_class_group] = copy.deepcopy(model.head)
        for c in self.classifier_pool:
            if c != None:
                for p in c.parameters():
                    p.requires_grad = False

        cur_adapters = model.get_adapter()
        self.cur_adapters = self.flatten_parameters(cur_adapters)
        vector = self.cur_adapters - self.prev_adapters
        # if task_id>0: #? 1
        self.adapter_vec.append(vector)
        self.adapter_vec_label.append(self.task_type)
        self.cluster_adapters()

    def train_and_evaluate(self, model: torch.nn.Module, criterion, data_loader: Iterable,
                           optimizer: torch.optim.Optimizer,
                           lr_scheduler, device: torch.device, class_mask=None, args=None, ):
        self.replay_use_count = np.zeros(self.num_classes, dtype=np.int64)
        self.drift_ema_task_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.drift_ema_task_cnt = np.zeros(self.num_classes, dtype=np.int64)
        self.drift_anchor_task_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.drift_anchor_task_cnt = np.zeros(self.num_classes, dtype=np.int64)

        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        ema_model = None
        for task_id in range(args.num_tasks):
            if task_id >= 1:
                self.replay_use_count = np.zeros(self.num_classes, dtype=np.int64)
                self.drift_ema_task_sum = np.zeros(self.num_classes, dtype=np.float64)
                self.drift_ema_task_cnt = np.zeros(self.num_classes, dtype=np.int64)
                self.drift_anchor_task_sum = np.zeros(self.num_classes, dtype=np.float64)
                self.drift_anchor_task_cnt = np.zeros(self.num_classes, dtype=np.int64)

            if task_id > 0 and args.reinit_optimizer:
                optimizer = create_optimizer(args, model)

            if task_id == 1 and len(args.adapt_blocks) > 0:
                ema_model = ModelEmaV2(model.get_adapter(), decay=args.ema_decay, device=device)
            model, optimizer = self.pre_train_task(
                model, data_loader[task_id]['train'], device, task_id, args
            )
            self._init_drift_centroids_for_old_classes(model, device, task_id, class_mask)

            for epoch in range(args.epochs):
                model = self.pre_train_epoch(model=model, epoch=epoch, task_id=task_id, args=args)
                train_stats = self.train_one_epoch(model=model,
                    criterion=criterion,
                    data_loader=data_loader[task_id]['train'],
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    max_norm=args.clip_grad,
                    set_training_mode=True,
                    task_id=task_id,
                    class_mask=class_mask,
                    ema_model=ema_model,
                    args=args,
                )
                if lr_scheduler:
                    lr_scheduler.step(epoch)

                self.update_epoch_drift(model=model,device=device,task_id=task_id,epoch=epoch,)

            self.post_train_task(model, task_id=task_id)
            if self.args.d_threshold:
                self.label_train_count[self.current_classes] += 1

            test_stats = self.evaluate_till_now(
                model=model,
                data_loader=data_loader,
                device=device,
                task_id=task_id,
                class_mask=class_mask,
                acc_matrix=acc_matrix,
                ema_model=ema_model,
                args=args,
            )

            if task_id >= 1:
                replay_dict = {
                    int(c): int(cnt)
                    for c, cnt in enumerate(self.replay_use_count.tolist())
                    if cnt > 0
                }
                record = {
                    "time": datetime.datetime.now().isoformat(),
                    "type": "replay_count",
                    "task": int(task_id),
                    "per_class_replay_count": replay_dict,
                }
                self._append_json_record(record)
                if task_id >= 1:
                    cls_order = list(replay_dict.keys())

                    dift_ema = {}
                    dift_anchor = {}
                    for c in cls_order:
                        idx = int(c)
                        cnt_e = int(self.drift_ema_task_cnt[idx])
                        cnt_a = int(self.drift_anchor_task_cnt[idx])
                        if cnt_e > 0:
                            val_e = float(self.drift_ema_task_sum[idx] / max(cnt_e, 1))
                        else:
                            val_e = 0.0

                        if cnt_a > 0:
                            val_a = float(self.drift_anchor_task_sum[idx] / max(cnt_a, 1))
                        else:
                            val_a = 0.0

                        dift_ema[idx] = val_e
                        dift_anchor[idx] = val_a

                    record_ema = {
                        "time": datetime.datetime.now().isoformat(),
                        "type": "drift amount",
                        "task": int(task_id),
                        "dift_ema": dift_ema,
                    }
                    self._append_json_record(record_ema)

                    record_anchor = {
                        "time": datetime.datetime.now().isoformat(),
                        "type": "drift amount",
                        "task": int(task_id),
                        "dift_anchor": dift_anchor,
                    }
                    self._append_json_record(record_anchor)
            self.log_drift_and_replay_stats(task_id=task_id, args=args)

            if args.output_dir and utils.is_main_process():
                Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

                checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
                state_dict = {'model': model.state_dict(),'ema_model': ema_model.state_dict() if ema_model is not None else None,'optimizer': optimizer.state_dict(),'epoch': epoch,'args': args,}
                if args.sched is not None and args.sched != 'constant':
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()

                utils.save_on_master(state_dict, checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},**{f'test_{k}': v for k, v in test_stats.items()},'task': task_id,}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        return acc_matrix

