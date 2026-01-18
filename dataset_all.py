import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from continual_datasets.continual_datasets import *
import utils


# =============================================================================
# 工具类与函数 (From dataset_o/dataset_all)
# =============================================================================

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes

    def __call__(self, img):
        return self.lambd(img, self.nb_classes)


def target_transform(x, nb_classes):
    return x + nb_classes


def build_transform(is_train, args):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    return transform


def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.task_inc:
        mode = 'til'
    elif args.domain_inc:
        mode = 'dil'
    elif args.versatile_inc:
        mode = 'vil'
    elif args.joint_train:
        mode = 'joint'
    else:
        mode = 'cil'

    # --------------------------------------------------------------------------
    # TIL / CIL 模式
    # --------------------------------------------------------------------------
    if mode in ['til', 'cil']:
        if 'iDigits' in args.dataset:
            # [修正] 使用 dataset_o.py 的正确逻辑，修复缩进和拼接顺序
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()

            # 1. 循环加载并切分所有数据集
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                dataset_train = AddDomain(dataset_train, i)
                dataset_val = AddDomain(dataset_val, i)

                splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
                mask.append(class_mask)

                # 将切分后的 task 数据暂存到线性列表中
                for k in range(len(splited_dataset)):
                    train.append(splited_dataset[k][0])
                    val.append(splited_dataset[k][1])

            # 2. [关键修复] 在循环外进行数据重组 (Interleave tasks across domains)
            splited_dataset = list()
            for i in range(args.num_tasks):
                # 取出每个数据集的第 i 个 task 进行合并
                t = [train[i + args.num_tasks * j] for j in range(len(dataset_list))]
                v = [val[i + args.num_tasks * j] for j in range(len(dataset_list))]
                splited_dataset.append((torch.utils.data.ConcatDataset(t), torch.utils.data.ConcatDataset(v)))

            args.nb_classes = len(splited_dataset[0][1].datasets[0].dataset.classes)
            class_mask = np.unique(np.array(mask), axis=0).tolist()[0]

        else:
            # 其他数据集 (DomainNet, CORe50 等) 使用标准逻辑
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
            args.nb_classes = len(dataset_val.classes)

    # --------------------------------------------------------------------------
    # DIL / VIL 模式
    # --------------------------------------------------------------------------
    elif mode in ['dil', 'vil']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            splited_dataset = list()

            for i in range(len(dataset_list)):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset_list[i],
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                dataset_train = AddDomain(dataset_train, i)
                dataset_val = AddDomain(dataset_val, i)
                # [同步] 移除 DomainWrapper，保持与 dataset_o 一致
                splited_dataset.append((dataset_train, dataset_val))

            args.nb_classes = len(dataset_val.classes)

        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            if args.dataset in ['CORe50']:
                splited_dataset = [(dataset_train[i], dataset_val) for i in range(len(dataset_train))]
                args.nb_classes = len(dataset_val.classes)
            else:
                splited_dataset = [(dataset_train[i], dataset_val[i]) for i in range(len(dataset_train))]
                args.nb_classes = len(dataset_val[0].classes)

    # --------------------------------------------------------------------------
    # Joint 模式
    # --------------------------------------------------------------------------
    elif mode in ['joint']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                dataset_train = AddDomain(dataset_train, i)
                dataset_val = AddDomain(dataset_val, i)
                train.append(dataset_train)
                val.append(dataset_val)
                args.nb_classes = len(dataset_val.classes)

            dataset_train = torch.utils.data.ConcatDataset(train)
            dataset_val = torch.utils.data.ConcatDataset(val)
            splited_dataset = [(dataset_train, dataset_val)]

            class_mask = None

        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset = [(dataset_train, dataset_val)]

            args.nb_classes = len(dataset_val.classes)
            class_mask = None

    else:
        raise ValueError(f'Invalid mode: {mode}')

    # --------------------------------------------------------------------------
    # VIL / RIL 场景构建 (保留 dataset_all 的 RIL 扩展能力)
    # --------------------------------------------------------------------------
    if args.versatile_inc:
        splited_dataset, class_mask, domain_list, args = build_vil_scenario(splited_dataset, args)

        # [保留] 如果 args 中开启了 RIL (Random IL)，则调用 RIL 构建器
        if getattr(args, 'random_inc', False):
            assert args.n_tasks is not None, "RIL requires --n_tasks"
            splited_dataset, class_mask, domain_list = build_ril_scenario(
                splited_dataset, class_mask, domain_list, args.n_tasks, getattr(args, 'shuffle', True)
            )
            args.num_tasks = args.n_tasks

        for c, d in zip(class_mask, domain_list):
            print(c, d)
    else:
        # dataset_o 不生成 domain_list，但为了接口一致性，这里生成默认值
        domain_list = [0 for _ in range(len(splited_dataset))]

    # --------------------------------------------------------------------------
    # 构建 DataLoaders
    # --------------------------------------------------------------------------
    for i in range(len(splited_dataset)):
        dataset_train, dataset_val = splited_dataset[i]

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask, domain_list



def get_dataset(dataset, transform_train, transform_val, mode, args):
    """
    完全复用 dataset_o.py 的实现，使用标准库类，避免使用自定义 CIL 类导致的差异。
    """
    if dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'CORe50':
        # [回退] 使用 standard CORe50 类，而非 CORe50CIL
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'DomainNet':
        # [回退] 使用 standard DomainNet 类，而非 DomainNetCIL
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)

    else:
        raise ValueError('Dataset {} not found.'.format(dataset))

    return dataset_train, dataset_val


def split_single_dataset(dataset_train, dataset_val, args):
    """
    复用 dataset_o.py 的实现。
    """
    # 兼容 ConcatDataset (防止 TIL 与 VIL 混合时出错)
    if hasattr(dataset_val, "classes"):
        nb_classes = len(dataset_val.classes)
    elif isinstance(dataset_val, torch.utils.data.ConcatDataset):
        first_subset = dataset_val.datasets[0]
        if hasattr(first_subset, "dataset") and hasattr(first_subset.dataset, "classes"):
            nb_classes = len(first_subset.dataset.classes)
        else:
            nb_classes = 10  # Fallback for iDigits
    else:
        nb_classes = 10

    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]

    split_datasets = list()
    mask = list()

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks):
        train_split_indices = list()
        test_split_indices = list()

        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        # 兼容 target 获取
        targets_train = dataset_train.targets if hasattr(dataset_train, 'targets') else dataset_train.labels
        targets_val = dataset_val.targets if hasattr(dataset_val, 'targets') else dataset_val.labels

        for k in range(len(targets_train)):
            if int(targets_train[k]) in scope:
                train_split_indices.append(k)

        for h in range(len(targets_val)):
            if int(targets_val[h]) in scope:
                test_split_indices.append(h)

        subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])

    return split_datasets, mask


def build_vil_scenario(splited_dataset, args):
    """
    复用 dataset_o.py 的逻辑，确保 VIL 场景构建一致。
    """
    datasets = list()
    class_mask = list()
    domain_list = list()

    for i in range(len(splited_dataset)):
        dataset, mask = split_single_dataset(splited_dataset[i][0], splited_dataset[i][1], args)
        datasets.append(dataset)
        class_mask.append(mask)
        for _ in range(len(dataset)):
            domain_list.append(f'D{i}')

    splited_dataset = sum(datasets, [])
    class_mask = sum(class_mask, [])

    args.num_tasks = len(splited_dataset)

    zipped = list(zip(splited_dataset, class_mask, domain_list))
    random.shuffle(zipped)
    splited_dataset, class_mask, domain_list = zip(*zipped)

    return splited_dataset, class_mask, domain_list, args


# =============================================================================
# 扩展功能 (保留自 dataset_all.py，供 RIL 使用)
# =============================================================================

def build_ril_scenario(splited_dataset, class_mask, domain_list, n_tasks, shuffle=True):
    zipped = list(zip(splited_dataset, class_mask, domain_list))
    if shuffle:
        random.shuffle(zipped)

    base = len(zipped) // n_tasks
    rem = len(zipped) % n_tasks

    new_dataset, new_mask, new_domain = [], [], []

    idx = 0
    for i in range(n_tasks):
        size = base + (1 if i < rem else 0)
        batch = zipped[idx:idx + size]
        idx += size

        train_sets = [b[0][0] for b in batch]
        val_sets = [b[0][1] for b in batch]
        merged_train = torch.utils.data.ConcatDataset(train_sets)
        merged_val = torch.utils.data.ConcatDataset(val_sets)

        new_dataset.append((merged_train, merged_val))
        # new_mask.append(sum([b[1] for b in batch], []))
        merged = []
        for b in batch:
            merged.extend(b[1])

        # 保序去重（Python 3.7+ dict 保序）
        merged = list(dict.fromkeys(int(x) for x in merged))

        new_mask.append(merged)
        new_domain.append([b[2] for b in batch])
    return new_dataset, new_mask, new_domain


def export_task_plan_txt(dataloader, out_path="task_plan.txt"):
    """
    保留用于调试和查看任务序列的工具
    """

    def _domain_name_from_ds(ds):
        cname = ds.__class__.__name__
        mapping = {
            "MNIST_RGB": "MNIST",
            "SVHN": "SVHN",
            "MNISTM": "MNISTM",
            "SynDigit": "SynDigit",
        }
        if cname in mapping: return mapping[cname]
        return cname

    def _classes_in_subset(sub):
        base = sub.dataset if isinstance(sub, Subset) else sub
        targets = base.targets if hasattr(base, 'targets') else base.labels
        idxs = getattr(sub, "indices", range(len(targets)))
        uniq = sorted({int(targets[i]) for i in idxs})
        classes = getattr(base, "classes", None)
        if classes is None: return [str(x) for x in uniq]
        return [str(classes[c]) for c in uniq]

    with open(out_path, "w", encoding="utf-8") as f:
        for t, pack in enumerate(dataloader):
            f.write(f"task{t}：\n")
            ds_val = pack["val"].dataset

            if isinstance(ds_val, ConcatDataset):
                for pv in ds_val.datasets:
                    base = pv.dataset if isinstance(pv, Subset) else pv
                    domain = _domain_name_from_ds(base)
                    for name in _classes_in_subset(pv):
                        f.write(f"{name}，{domain}\n")
            else:
                base = ds_val.dataset if isinstance(ds_val, Subset) else ds_val
                domain = _domain_name_from_ds(base)
                for name in _classes_in_subset(ds_val):
                    f.write(f"{name}，{domain}\n")

class AddDomain(Dataset):
    """
    Wrap a dataset so that __getitem__ returns (x, y, domain_id).
    Also forwards common attributes (targets/labels/classes) used by split_single_dataset/export_task_plan_txt.
    """
    def __init__(self, base, domain_id: int):
        self.base = base
        self.domain_id = int(domain_id)

        # forward commonly-used attributes
        if hasattr(base, "targets"):
            self.targets = base.targets
        if hasattr(base, "labels"):
            self.labels = base.labels
        if hasattr(base, "classes"):
            self.classes = base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]   # expected (img, label)
        return x, y, torch.tensor(self.domain_id, dtype=torch.long)

    def __getattr__(self, name):
        # fallback to wrapped dataset
        return getattr(self.base, name)
