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
# [新增] 引入 PrototypeManager
from prototype_manager import PrototypeManager


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

        # [新增] SCE / Prototype Manager 初始化
        # 仅初始化对象，不影响原有逻辑流程
        feature_dim = model.embed_dim if hasattr(model, 'embed_dim') else 768
        self.proto_manager = PrototypeManager(feature_dim, self.num_classes, device)
        self.use_sce = args.use_sce if hasattr(args, 'use_sce') else False

        # --- 初始化变量 (保留结构以防报错，但不再用于日志) ---
        self.class_seen = np.zeros(self.num_classes, dtype=np.int64)
        self.cum_counts_total = np.zeros(self.num_classes, dtype=np.int64)
        self.forget_score = np.zeros(self.num_classes, dtype=np.float64)
        self.class_feat_init = [None for _ in range(self.num_classes)]
        self.class_drift_hist = [
            deque(maxlen=int(getattr(self.args, 'drift_hist_len', 10)))
            for _ in range(self.num_classes)
        ]
        self.class_drift_score = np.zeros(self.num_classes, dtype=np.float64)
        self.old_classes = []
        self.drift_anchor_task_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.drift_anchor_task_cnt = np.zeros(self.num_classes, dtype=np.int64)
        self.replay_buffer = {}

        self.label2task = {}
        for _t, _mask in enumerate(class_mask):
            for _c in _mask:
                _ci = int(_c)
                if _ci not in self.label2task:
                    self.label2task[_ci] = _t
        self.common_direction = None

    @torch.no_grad()
    def update_epoch_drift(self, model, device, task_id: int, epoch: int):
        return

    @torch.no_grad()
    def _init_drift_centroids_for_old_classes(self, model, device, task_id, class_mask):
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
        printed_warning = False

        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                _, targets, domains = batch[:3]
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                if not printed_warning:
                    print(
                        "[WARN] _pair_counts_from_loader: Batch length is 2. Domains missing! Defaulting to domain 0.")
                    printed_warning = True
                _, targets = batch
                domains = torch.zeros_like(targets)
            else:
                continue

            t = targets.view(-1).to('cpu')
            d = domains.view(-1).to('cpu')

            for ci, di in zip(t, d):
                c_idx, d_idx = int(ci), int(di)
                if 0 <= c_idx < counts.shape[0] and 0 <= d_idx < counts.shape[1]:
                    counts[c_idx, d_idx] += 1

        return counts

    def kl_div(self, p, q):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        eps = 1e-8
        kl = torch.mean(torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=1))
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
            for batch_idx, batch in enumerate(data_loader):
                if self.args.develop and batch_idx > 200:
                    break

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    input, target, _ = batch
                else:
                    input, target = batch

                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(input)

                if output.shape[-1] > self.num_classes:
                    output, _, _ = self.get_max_label_logits(output, self.current_classes, task_id=self.current_task,
                                                             slice=True)

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

    # def detect_labels_to_be_added(self, inference_acc, thresholds=[]):
    #     labels_with_low_accuracy = []
    #
    #     if self.args.d_threshold:
    #         triplets = zip(self.current_classes, inference_acc, thresholds)
    #     else:
    #         triplets = ((l, a, self.args.thre) for l, a in zip(self.current_classes, inference_acc))
    #
    #     for label, acc, thre in triplets:
    #         cond_acc = (acc <= thre)
    #         if cond_acc:
    #             labels_with_low_accuracy.append(label)
    #
    #     print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
    #     return labels_with_low_accuracy
    def detect_labels_to_be_added(self, inference_acc, thresholds=None):
        labels_with_low_accuracy = []

        use_passed = (
                thresholds is not None
                and isinstance(thresholds, (list, tuple))
                and len(thresholds) == len(self.current_classes)
        )

        if use_passed:
            triplets = zip(self.current_classes, inference_acc, thresholds)
        else:
            triplets = ((l, a, self.args.thre) for l, a in zip(self.current_classes, inference_acc))

        for label, acc, thre in triplets:
            if acc <= thre:
                labels_with_low_accuracy.append(label)

        print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
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

            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                input, target, domain = batch
                domain = domain.to(device, non_blocking=True).long()
            else:
                input, target = batch
                domain = None

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # --- Forward ---
            output = model(input)

            # Distillation logic (retained)
            distill_loss = 0
            if self.distill_head is not None:
                feature = model.forward_features(input)[:, 0]
                output_distill = self.distill_head(feature)
                # mask_nodes = torch.isin(torch.tensor(self.labels_in_head), torch.tensor(self.current_classes))
                # cur_class_nodes = torch.where(mask_nodes)[0]
                # m_added = torch.isin(torch.tensor(self.labels_in_head[cur_class_nodes]),
                #                      torch.tensor(list(self.added_classes_in_cur_task)))
                # distill_node_indices = self.labels_in_head[cur_class_nodes][~m_added]
                # distill_loss = self.kl_div(output[:, distill_node_indices], output_distill[:, distill_node_indices])
                labels_in_head_t = torch.as_tensor(self.labels_in_head, device=output.device, dtype=torch.long)
                current_classes_t = torch.as_tensor(self.current_classes, device=output.device, dtype=torch.long)

                # ✅ cur_class_nodes 是 node column indices（logits 的列索引）
                mask_nodes = torch.isin(labels_in_head_t, current_classes_t)
                cur_class_nodes = torch.where(mask_nodes)[0]

                # 排除当前任务新加的那些类对应的 node（避免蒸馏“新节点”）
                if len(self.added_classes_in_cur_task) > 0:
                    added_t = torch.as_tensor(list(self.added_classes_in_cur_task), device=output.device,
                                              dtype=torch.long)
                    m_added = torch.isin(labels_in_head_t[cur_class_nodes], added_t)
                else:
                    m_added = torch.zeros_like(cur_class_nodes, dtype=torch.bool)

                distill_cols = cur_class_nodes[~m_added]  # ✅ 仍然是 node indices

                distill_loss = 0
                if distill_cols.numel() > 0:
                    distill_loss = self.kl_div(output[:, distill_cols], output_distill[:, distill_cols])

            # --- Logits Masking & Slicing ---
            if output.shape[-1] > self.num_classes:
                output, _, _ = self.get_max_label_logits(output, class_mask[task_id], slice=False)
                if len(self.added_classes_in_cur_task) > 0:
                    for added_class in self.added_classes_in_cur_task:
                        cur_node = np.where(self.labels_in_head == added_class)[0][-1]
                        output[:, added_class] = output[:, cur_node]
                output = output[:, :self.num_classes]

            logits_full = output
            logits_new = logits_full
            target_new = target

            def _mask_logits(logits_x, allow_set):
                if logits_x is None:
                    return None
                allow = np.array(sorted(list(allow_set)), dtype=np.int64)
                all_ids = np.arange(self.num_classes, dtype=np.int64)
                not_mask = np.setdiff1d(all_ids, allow)
                if not_mask.size == 0:
                    return logits_x
                not_mask = torch.tensor(not_mask, dtype=torch.int64, device=logits_x.device)
                return logits_x.index_fill(dim=1, index=not_mask, value=float('-inf'))

            if args.train_mask and class_mask is not None:
                cur_task_ids = set(int(x) for x in class_mask[task_id])

                if logits_new is not None:
                    allow_new = cur_task_ids | set(int(x) for x in torch.unique(target_new).detach().cpu().tolist())
                    logits_new = _mask_logits(logits_new, allow_new)

                allow_all = cur_task_ids | set(int(x) for x in torch.unique(target).detach().cpu().tolist())
                logits = _mask_logits(logits_full, allow_all)
            else:
                logits = logits_full

            use_la_train = bool(getattr(args, 'use_la_train', False))

            if use_la_train:
                counts = torch.tensor(np.maximum(self.cum_counts_total, 1),
                                      dtype=torch.float32, device=device)
                pi = counts / counts.sum()
                la_tau = float(getattr(args, 'la_tau', 1.0))
                logits_la = logits_new - la_tau * torch.log(pi).unsqueeze(0)
                ce_loss_new = F.cross_entropy(logits_la, target_new)
            else:
                ce_loss_new = F.cross_entropy(logits_new, target_new)

            loss = ce_loss_new
            if self.args.IC:
                alpha = float(getattr(args, "alpha", 1.0))
                if isinstance(distill_loss, torch.Tensor):
                    dl = float(distill_loss.detach().item())
                else:
                    dl = float(distill_loss)
                if dl > 0 and math.isfinite(dl):
                    loss = loss + alpha * distill_loss

            if self.args.use_cast_loss:
                if len(self.adapter_vec) > args.k:
                    cur_adapters = model.get_adapter()
                    self.cur_adapters = self.flatten_parameters(cur_adapters)
                    diff_adapter = self.cur_adapters - self.prev_adapters

                    if torch.isnan(diff_adapter).any() or torch.isinf(diff_adapter).any():
                        print(f"[Warning] NaN/Inf detected... Skipping CAST loss.")
                    else:
                        _, other = self.find_same_cluster_items(diff_adapter)
                        other_disentangled = []
                        if getattr(self, 'common_direction', None) is not None:
                            c_dir = self.common_direction
                            for o in other:
                                proj_val = torch.dot(o, c_dir)
                                o_shared = proj_val * c_dir
                                o_spec = o - o_shared
                                other_disentangled.append(o_spec)
                        else:
                            other_disentangled = other

                        weights = self.calculate_l2_distance(diff_adapter, other_disentangled)
                        sim = 0
                        for o_spec, w in zip(other_disentangled, weights):
                            if self.args.norm_cast:
                                norm_diff = torch.norm(diff_adapter) + 1e-8
                                norm_o = torch.norm(o_spec) + 1e-8
                                sim += w * torch.matmul(diff_adapter, o_spec) / (norm_diff * norm_o)
                            else:
                                sim += w * torch.matmul(diff_adapter, o_spec)

                        orth_loss = args.beta * torch.abs(sim)
                        if orth_loss > 0:
                            loss += orth_loss

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            found_nan = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        found_nan = True
                        break
            if found_nan:
                print(f"[Warning] NaN/Inf gradient detected at step {batch_idx}. Skipping optimization step.")
                optimizer.zero_grad()
                continue

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

        def _agg(x, idx, label):
            if len(idx) == 1:
                return x[:, idx[0]]
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
            for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop and batch_idx > 20:
                    break

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    input, target, _ = batch
                else:
                    input, target = batch

                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                logits_main = model(input)
                logits_main, _, _ = self.get_max_label_logits(logits_main, class_mask[task_id], task_id=task_id,
                                                              target=target, slice=True)

                outputs_for_ensemble = [logits_main.softmax(dim=1)]

                if ema_model is not None:
                    tmp_adapter = model.get_adapter()
                    model.put_adapter(ema_model.module)

                    logits_ema = model(input)
                    logits_ema, _, _ = self.get_max_label_logits(
                        logits_ema, class_mask[task_id], slice=True
                    )

                    outputs_for_ensemble.append(logits_ema.softmax(dim=1))
                    model.put_adapter(tmp_adapter)

                final_output = torch.stack(outputs_for_ensemble, dim=-1).max(dim=-1)[0]
                loss = criterion(final_output, target)

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
                        minlength=self.num_classes, )
                    per_class_total += binc_total
                    per_class_correct += binc_correct.astype(np.int64)
                except Exception:
                    pass

            if total_sum > 0:
                print(f"Max Pooling acc: {correct_sum / total_sum}")
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

            print(f"[DEBUG] Eval Task {task_id}, Cur Task {self.current_task}, d_thresh {self.args.d_threshold}")
            if getattr(self.args, "d_threshold", False) and task_id == (self.current_task - 1):
                domain_idx = int(self.label_train_count[self.current_classes][0])
                if domain_idx < self.acc_per_label.shape[1]:
                    self.acc_per_label[self.current_classes, domain_idx] += np.round(
                        per_class_acc[self.current_classes], decimals=3
                    )

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

        avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
        diagonal = np.diag(acc_matrix)
        result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
            task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2]
        )

        forgetting = 0.0
        backward = 0.0
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
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
                col_mean = np.nanmean(np.where(mask, np.nan, self.adapter_vec_array), axis=0)
                rows, cols = np.where(mask)
                self.adapter_vec_array[rows, cols] = col_mean[cols]
                print("[CAST] Imputed NaN/Inf in adapter_vec_array with column means.")

            mean_vec = np.mean(self.adapter_vec_array, axis=0)
            mean_norm = np.linalg.norm(mean_vec)
            if mean_norm > 1e-8:
                self.common_direction = torch.tensor(mean_vec / mean_norm, dtype=torch.float32).to(self.device)
            else:
                self.common_direction = torch.zeros_like(torch.tensor(mean_vec)).to(self.device)
            print(f"[D-CAST] Common Direction computed. Norm: {mean_norm:.4f}")

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

    def pre_train_task(self, model, loader_dict, device, task_id, args):
        self.current_task += 1
        self.current_class_group = int(min(self.class_mask[task_id]) / self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]
        train_loader = loader_dict['train']

        with torch.no_grad():
            counts = self._class_counts_from_loader(train_loader)
        self.cur_class_counts = counts.detach().cpu().numpy().astype(np.int64)
        if getattr(self, 'cum_counts_total', None) is None:
            self.cum_counts_total = np.zeros(self.num_classes, dtype=np.int64)
        self.cum_counts_total[:len(self.cur_class_counts)] += self.cur_class_counts

        print(f"\n\nTASK : {task_id}")
        self.added_classes_in_cur_task = set()

        # ---------- [新增] 新类强制扩 1 个节点（只对 task_id > 0） ----------
        force_new_labels = []
        if task_id > 0:
            for c in self.current_classes:
                ci = int(c)
                if self.label2task.get(ci, -1) == task_id:
                    force_new_labels.append(ci)
            if len(force_new_labels) > 0:
                print(f"[IC] Force expand new classes in task {task_id}: {force_new_labels}")

        # 将“阈值触发扩容”的结果先收集到这里，最后统一扩一次
        labels_to_be_added = []

        if self.class_group_train_count[self.current_class_group] == 0:
            self.distill_head = None
            # [保留] SCE 初始化逻辑，包裹在 use_sce 中
            if self.use_sce:
                print("[SCE] First time seeing this group. Initializing prototypes.")
                init_thresholds = [self.args.thre] * len(self.current_classes)
                self.proto_manager.get_drift_based_thresholds(
                    model, train_loader, self.current_classes, init_thresholds, sensitivity=args.sce_sensitivity
                )
        else:  # already seen classes
            if self.args.IC:
                self.distill_head = self.classifier_pool[self.current_class_group]
                print("[IC] Computing thresholds using Train Set.")
                inf_acc_list = self.inference_acc(model, train_loader, device)

                print(f"\n[DEBUG] Pre-Task Inference Accuracy (Task {task_id}):")
                for i, label in enumerate(self.current_classes):
                    acc_val = inf_acc_list[i]
                    offset_msg = ""
                    # 仅在开启 d_threshold 且有历史数据时计算 Offset (历史平均 - 当前)
                    if self.args.d_threshold and self.class_group_train_count[self.current_class_group] > 0:
                        count = self.class_group_train_count[self.current_class_group]
                        hist_avg = np.sum(self.acc_per_label[label, :count]) / count
                        offset = hist_avg - acc_val
                        offset_msg = f" | Offset: {offset:.4f} (Hist: {hist_avg:.4f})"
                    print(f"  Class {label:<4}: Acc = {acc_val:.4f}{offset_msg}")

                thresholds = []
                if self.args.d_threshold:
                    count = self.class_group_train_count[self.current_class_group]
                    if count > 0:
                        average_accs = np.sum(self.acc_per_label[self.current_classes, :count], axis=1) / count
                    else:
                        average_accs = np.zeros_like(inf_acc_list)

                    # [修正]: 仅对历史平均准确率 > 0 的旧类应用遗忘公式；新类直接用默认阈值
                    default_thre = float(getattr(self.args, 'thre', 0.0))
                    raw_thresholds = np.full_like(average_accs, default_thre, dtype=np.float64)

                    mask_old = (average_accs > 0)
                    if np.any(mask_old):
                        old_avg = average_accs[mask_old]
                        old_inf = np.array(inf_acc_list)[mask_old]
                        dynamic_vals = self.args.gamma * (old_avg - old_inf) / old_avg
                        dynamic_vals_t = torch.tensor(dynamic_vals)
                        dynamic_vals_tanh = self.tanh(dynamic_vals_t).numpy()
                        raw_thresholds[mask_old] = dynamic_vals_tanh

                    # 最终阈值列表化（应用 max(calc, self.args.thre) 的逻辑保持不变）
                    thresholds = []
                    for t_val in raw_thresholds:
                        if t_val > self.args.thre:
                            thresholds.append(round(float(t_val), 2))
                        else:
                            thresholds.append(self.args.thre)

                # [保留] SCE Drift 检测逻辑
                if self.use_sce:
                    if not thresholds:
                        thresholds = [self.args.thre] * len(self.current_classes)
                    print("[SCE] Detecting semantic drift for dynamic expansion.")
                    thresholds = self.proto_manager.get_drift_based_thresholds(
                        model, train_loader, self.current_classes, thresholds, sensitivity=args.sce_sensitivity
                    )

                labels_to_be_added = self.detect_labels_to_be_added(
                    inf_acc_list,
                    thresholds,
                )

                formatted_str = ", ".join(f"{t:.8f}" for t in thresholds) if thresholds else ""
                if thresholds:
                    print(
                        f"Thresholds for class {self.current_classes[0]}~{self.current_classes[-1]} : [{formatted_str}]")

        # ---------- [新增] 统一扩容：阈值触发 + 新类强制扩（合并去重，只扩一次） ----------
        if self.args.IC:
            merged = list(dict.fromkeys([int(x) for x in (list(labels_to_be_added) + list(force_new_labels))]))
            if len(merged) > 0:
                new_head = self.set_new_head(
                    model,
                    np.array(merged, dtype=np.int64),
                    task_id
                ).to(device)
                model.head = new_head

        optimizer = create_optimizer(args, model)

        with torch.no_grad():
            prev_adapters = model.get_adapter()
            self.prev_adapters = self.flatten_parameters(prev_adapters)
            self.prev_adapters.requires_grad = False

        if task_id == 0:
            self.task_type_list.append("Initial")
            return model, optimizer

        prev_class = self.class_mask[task_id - 1]
        cur_class = self.class_mask[task_id]
        self.cur_domain = self.domain_list[task_id]
        self.task_type = "DIL" if (prev_class == cur_class) else "CIL"
        self.task_type_list.append(self.task_type)
        print(f"Current task : {self.task_type}")

        return model, optimizer

        prev_class = self.class_mask[task_id - 1]
        cur_class = self.class_mask[task_id]
        self.cur_domain = self.domain_list[task_id]
        self.task_type = "DIL" if (prev_class == cur_class) else "CIL"
        self.task_type_list.append(self.task_type)
        print(f"Current task : {self.task_type}")

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
        self.adapter_vec.append(vector)
        self.adapter_vec_label.append(self.task_type)
        self.cluster_adapters()

    def train_and_evaluate(self, model: torch.nn.Module, criterion, data_loader: Iterable,
                           optimizer: torch.optim.Optimizer,
                           lr_scheduler, device: torch.device, class_mask=None, args=None, ):
        self.drift_ema_task_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.drift_ema_task_cnt = np.zeros(self.num_classes, dtype=np.int64)
        self.drift_anchor_task_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.drift_anchor_task_cnt = np.zeros(self.num_classes, dtype=np.int64)

        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        ema_model = None
        for task_id in range(args.num_tasks):
            if task_id >= 1:
                self.drift_ema_task_sum = np.zeros(self.num_classes, dtype=np.float64)
                self.drift_ema_task_cnt = np.zeros(self.num_classes, dtype=np.int64)
                self.drift_anchor_task_sum = np.zeros(self.num_classes, dtype=np.float64)
                self.drift_anchor_task_cnt = np.zeros(self.num_classes, dtype=np.int64)

            if task_id > 0 and args.reinit_optimizer:
                optimizer = create_optimizer(args, model)

            if task_id == 1 and len(args.adapt_blocks) > 0:
                ema_model = ModelEmaV2(model.get_adapter(), decay=args.ema_decay, device=device)

            model, optimizer = self.pre_train_task(
                model, data_loader[task_id], device, task_id, args
            )

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

            self.post_train_task(model, task_id=task_id)
            if self.args.d_threshold:
                self.label_train_count[self.current_classes] += 1

            if args.output_dir and utils.is_main_process():
                Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

                checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
                state_dict = {'model': model.state_dict(),
                              'ema_model': ema_model.state_dict() if ema_model is not None else None,
                              'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args, }
                if args.sched is not None and args.sched != 'constant':
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()

                utils.save_on_master(state_dict, checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}, 'task': task_id, }
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        return acc_matrix