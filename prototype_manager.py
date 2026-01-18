import torch
import torch.nn.functional as F
import numpy as np


class PrototypeManager:
    """
    语义一致性管理器 (SCE)
    """

    def __init__(self, feature_dim, num_classes, device, momentum=0.9):
        self.device = device
        self.num_classes = num_classes
        self.momentum = momentum

        # 存储每个类别的全局原型
        self.global_prototypes = torch.zeros(num_classes, feature_dim).to(device)
        self.has_initialized = torch.zeros(num_classes, dtype=torch.bool).to(device)

    def compute_current_prototypes(self, model, data_loader, current_classes):
        model.eval()
        # 初始化 sum 和 count
        sums = torch.zeros(self.num_classes, self.global_prototypes.shape[1]).to(self.device)
        counts = torch.zeros(self.num_classes).to(self.device)

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2:
                    input, target = batch
                else:
                    input, target, _ = batch[0], batch[1], batch[2]

                input = input.to(self.device)
                target = target.to(self.device)

                features = model.forward_features(input)
                if len(features.shape) == 3:
                    features = features[:, 0]

                # 注意：这里会把所有域的数据聚合到同一个 c 下
                for c in current_classes:
                    mask = (target == c)
                    if mask.sum() > 0:
                        sums[c] += features[mask].sum(dim=0)
                        counts[c] += mask.sum()

        current_prototypes = sums / (counts.unsqueeze(1) + 1e-6)
        return current_prototypes, counts

    def get_drift_based_thresholds(self, model, data_loader, current_classes, base_thresholds, sensitivity=2.0):
        curr_protos, counts = self.compute_current_prototypes(model, data_loader, current_classes)
        adjusted_thresholds = []

        # [快照]：在循环开始前，锁定“旧类”的状态。
        # 这样即使循环中初始化了某个类，也不会影响后续重复项的判定。
        is_old_class_mask = self.has_initialized.clone()

        # [防重]：记录本轮已经更新过原型的类，防止因为列表重复(多域)导致双重Update
        updated_in_this_call = set()

        for i, c in enumerate(current_classes):
            base_thre = base_thresholds[i]

            # 判断依据：必须是“快照”里存在的旧类
            if is_old_class_mask[c]:
                # --- 旧类逻辑 (Old Class) ---
                p_old = F.normalize(self.global_prototypes[c].unsqueeze(0), dim=1)
                p_new = F.normalize(curr_protos[c].unsqueeze(0), dim=1)

                drift = 1.0 - torch.mm(p_old, p_new.t()).item()
                drift_factor = drift * sensitivity
                new_thre = min(1.0, base_thre + drift_factor)

                print(f"Class {c}: Drift={drift:.4f}, Base_Thre={base_thre:.2f} -> New_Thre={new_thre:.2f}")
                adjusted_thresholds.append(new_thre)

                # [关键修正]：如果该类在本轮已经更新过一次，就不要再更新了
                if counts[c] > 0 and c not in updated_in_this_call:
                    self.global_prototypes[c] = (self.momentum * self.global_prototypes[c]) + \
                                                ((1 - self.momentum) * curr_protos[c])
                    updated_in_this_call.add(c)  # 标记：这个类我已处理过更新

            else:
                # --- 新类逻辑 (New Class) ---
                # 无论是第几次遇到，只要快照里它是新类，就按新类处理
                adjusted_thresholds.append(base_thre)

                # 初始化：同样防止重复初始化
                if counts[c] > 0 and not self.has_initialized[c]:
                    self.global_prototypes[c] = curr_protos[c]
                    self.has_initialized[c] = True
                    print(f"Class {c}: New Class Initialized (Base_Thre={base_thre:.2f}). Skip Drift.")
                elif not self.has_initialized[c]:
                    # counts[c] == 0，无数据，跳过
                    pass

        return adjusted_thresholds