import random
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils


class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes

    def __call__(self, img):
        return self.lambd(img, self.nb_classes)


def target_transform(x, nb_classes):
    return x + nb_classes


def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)
    # domain_list = []

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

    if mode in ['til', 'cil']:
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

                splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
                mask.append(class_mask)

                for i in range(len(splited_dataset)):
                    train.append(splited_dataset[i][0])
                    val.append(splited_dataset[i][1])

            splited_dataset = list()
            for i in range(args.num_tasks):
                t = [train[i + args.num_tasks * j] for j in range(len(dataset_list))]
                v = [val[i + args.num_tasks * j] for j in range(len(dataset_list))]
                splited_dataset.append((torch.utils.data.ConcatDataset(t), torch.utils.data.ConcatDataset(v)))

            args.nb_classes = len(splited_dataset[0][1].datasets[0].dataset.classes)
            class_mask = np.unique(np.array(mask), axis=0).tolist()[0]

        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
            args.nb_classes = len(dataset_val.classes)

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

    # ⬇️ 插入 VIL/RIL 模式
    if args.versatile_inc:
        splited_dataset, class_mask, domain_list, args = build_vil_scenario(splited_dataset, args)

        if getattr(args, 'random_inc', False):
            assert args.n_tasks is not None, "RIL requires --n_tasks"
            splited_dataset, class_mask, domain_list = build_ril_scenario(
                splited_dataset, class_mask, domain_list, args.n_tasks, args.shuffle
            )
            args.num_tasks = args.n_tasks

        for c, d in zip(class_mask, domain_list):
            print(c, d)
    else:
        # ⭐ 非 VIL/RIL 情况（CIL / DIL / joint），给一个简单的占位 domain_list
        domain_list = [0 for _ in range(len(splited_dataset))]

    for i in range(len(splited_dataset)):
        # 1. 获取原始的 Train 和 Val(Test)
        full_dataset_train, dataset_val = splited_dataset[i]

        # 2. 🟢 新增：从训练集中划分 10% 作为内部验证集 (Inner Val)
        # 获取训练集总长度
        n_train = len(full_dataset_train)
        indices = list(range(n_train))

        # 即使 args.shuffle=False，为了划分的随机性，这里建议打乱索引
        # 如果需要严格复现，确保 seed 已固定 (main中已做)
        random.shuffle(indices)

        split_point = int(n_train * 0.9)  # 90% 用于训练
        train_indices = indices[:split_point]
        inner_val_indices = indices[split_point:]

        # 创建 Subset
        dataset_train = Subset(full_dataset_train, train_indices)
        dataset_inner_val = Subset(full_dataset_train, inner_val_indices)

        # 3. 构建 Sampler
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_inner_val = torch.utils.data.SequentialSampler(dataset_inner_val)  # 验证集不需要 shuffle
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        # 4. 构建 Loader
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        # 🟢 新增 loader
        data_loader_inner_val = torch.utils.data.DataLoader(
            dataset_inner_val, sampler=sampler_inner_val,
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

        # 🟢 返回字典包含三部分
        dataloader.append({
            'train': data_loader_train,
            'inner_val': data_loader_inner_val,
            'val': data_loader_val
        })

    return dataloader, class_mask, domain_list

def get_dataset(dataset, transform_train, transform_val, mode, args, ):
    if dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'CORe50':
        if mode == 'cil':
            dataset_train = CORe50CIL(args.data_path, train=True, transform=transform_train)
            dataset_val = CORe50CIL(args.data_path, train=False, transform=transform_val)
        else:
            dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
            dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'DomainNet':
        if mode in ['til', 'cil']:
            dataset_train = DomainNetCIL(args.data_path,train=True,download=True,transform=transform_train,)
            dataset_val = DomainNetCIL(args.data_path,train=False,download=True,transform=transform_val,)
        else:
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
    nb_classes = len(dataset_val.classes)
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

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)

        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)

        subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])

    return split_datasets, mask


def build_vil_scenario(splited_dataset, args):
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
def build_ril_scenario(splited_dataset, class_mask, domain_list, n_tasks, shuffle=True):
    """
    输入:
        - splited_dataset: List[Tuple[train_set, val_set]]，来自 VIL 模式的每个类-域组合
        - class_mask: List[List[int]]，对应每个组合的类别
        - domain_list: List[str]，对应每个组合的域
        - n_tasks: 最终 RIL 任务数
        - shuffle: 是否打乱组合顺序
    输出:
        - 新的 splited_dataset/class_mask/domain_list（长度为 n_tasks）
    """
    zipped = list(zip(splited_dataset, class_mask, domain_list))
    if shuffle:
        random.shuffle(zipped)

    base = len(zipped) // n_tasks
    rem = len(zipped) % n_tasks

    new_dataset, new_mask, new_domain = [], [], []

    idx = 0
    for i in range(n_tasks):
        size = base + (1 if i < rem else 0)
        batch = zipped[idx:idx+size]
        idx += size
        # 合并数据
        train_sets = [b[0][0] for b in batch]
        val_sets = [b[0][1] for b in batch]
        merged_train = torch.utils.data.ConcatDataset(train_sets)
        merged_val = torch.utils.data.ConcatDataset(val_sets)
        new_dataset.append((merged_train, merged_val))
        new_mask.append(sum([b[1] for b in batch], []))
        new_domain.append([b[2] for b in batch])
    return new_dataset, new_mask, new_domain

def export_task_plan_txt(dataloader, out_path="task_plan.txt"):
    """
    将每个 task 的精确 (类名, 域名) 映射写入文本文件。
    - 兼容 RIL：当 dataloader[task]['train'/'val'].dataset 为 ConcatDataset 时，
      逐个子 Subset 恢复其类集合，并从对应的 train 子块推断域名。
    - 兼容 DomainNet / iDigits / CORe50：
      * DomainNet: 直接从 ImageFolder 路径 .../train/<domain> 或 .../test/<domain> 取域名
        （其构造即逐域建立 ImageFolder）:contentReference[oaicite:1]{index=1}。
      * iDigits: 用数据集类名映射为域名（MNIST、SVHN、MNISTM、SynDigit）。
      * CORe50: 训练集按会话 s1/s2/... 组织（域名即会话名）:contentReference[oaicite:2]{index=2}。
    """
    import os
    from torch.utils.data import Subset, ConcatDataset

    # def _domain_name_from_ds(ds):
    #     # 优先用 ImageFolder 的 root 路径末段
    #     root = getattr(ds, "root", None)
    #     if isinstance(root, str):
    #         last = os.path.basename(root.rstrip("/"))
    #         parent = os.path.basename(os.path.dirname(root.rstrip("/")))
    #         # 兼容 .../train/<domain> 或 .../test/<domain>
    #         return parent if last in ("train", "test") else last
    #     # 回退：根据数据集类名映射（iDigits 等）
    #     cname = ds.__class__.__name__
    #     mapping = {
    #         "MNIST_RGB": "MNIST",
    #         "SVHN": "SVHN",
    #         "MNISTM": "MNISTM",
    #         "SynDigit": "SynDigit",
    #     }
    #     return mapping.get(cname, cname)

    def _domain_name_from_ds(ds):
        cname = ds.__class__.__name__
        mapping = {
            "MNIST_RGB": "MNIST",
            "SVHN": "SVHN",
            "MNISTM": "MNISTM",
            "SynDigit": "SynDigit",
        }
        if cname in mapping:
            return mapping[cname]

        if ds.__class__.__name__ == "DomainNetCIL":
            return "Domain_All"

        root = getattr(ds, "root", None)
        if isinstance(root, str):
            last = os.path.basename(root.rstrip("/"))
            parent = os.path.basename(os.path.dirname(root.rstrip("/")))
            return parent if last in ("train", "test") else last

        return cname

    def _classes_in_subset(sub):
        # sub: Subset 或 Dataset
        base = sub.dataset if isinstance(sub, Subset) else sub
        # 取标签向量
        if hasattr(base, "targets"):
            targets = base.targets
        elif hasattr(base, "labels"):
            targets = base.labels
        elif hasattr(base, "samples"):  # torchvision ImageFolder
            targets = [x[1] for x in getattr(base, "samples")]
        else:
            raise AttributeError("底层数据集缺少 targets/labels/samples，无法恢复类别。")
        idxs = getattr(sub, "indices", None)
        if idxs is None:
            idxs = range(len(targets))
        uniq = sorted({int(targets[i]) for i in idxs})  # 该子块真实出现的类ID集合（与 split 时一致）:contentReference[oaicite:3]{index=3}
        classes = getattr(base, "classes", None)
        if classes is None:
            return [str(x) for x in uniq]
        return [str(classes[c]) for c in uniq]

    with open(out_path, "w", encoding="utf-8") as f:
        for t, pack in enumerate(dataloader):
            f.write(f"task{t}：\n")
            ds_train = pack["train"].dataset
            ds_val = pack["val"].dataset

            # RIL: ConcatDataset 由若干 Subset 组成，顺序与 train 一致（二者均由同一批 batch 顺序拼接）:contentReference[oaicite:4]{index=4}
            if isinstance(ds_val, ConcatDataset):
                parts_val = list(ds_val.datasets)
                parts_train = list(ds_train.datasets) if isinstance(ds_train, ConcatDataset) else [ds_train] * len(parts_val)
                for pv, pt in zip(parts_val, parts_train):
                    base_train = pt.dataset if isinstance(pt, Subset) else pt
                    domain = _domain_name_from_ds(base_train)
                    for name in _classes_in_subset(pv):
                        f.write(f"{name}，{domain}\n")
            else:
                # 非 RIL：val 可能是 Subset 或单个 Dataset
                base_val = ds_val.dataset if isinstance(ds_val, Subset) else ds_val
                domain = _domain_name_from_ds(base_val)
                if isinstance(ds_val, Subset):
                    for name in _classes_in_subset(ds_val):
                        f.write(f"{name}，{domain}\n")
                else:
                    # 极少数 joint 情况：全量数据一个域
                    classes = getattr(base_val, "classes", [])
                    for name in classes:
                        f.write(f"{str(name)}，{domain}\n")

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

class DomainNetCIL(Dataset):

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.transform = transform
        self.train = train

        split = "train" if train else "test"
        split_root = os.path.join(root, "VIL_DomainNet", split)  # ⚠️ 这里按你实际路径改

        # 扫描所有域 & 类
        domains = sorted(d for d in os.listdir(split_root)
                         if os.path.isdir(os.path.join(split_root, d)))

        # 类名去重（同一个类在多个域出现，只保留一个）
        class_names = set()
        for d in domains:
            d_root = os.path.join(split_root, d)
            for cname in os.listdir(d_root):
                cdir = os.path.join(d_root, cname)
                if os.path.isdir(cdir):
                    class_names.add(cname)

        self.classes = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 建立完整的 (img_path, class_idx) 列表
        samples = []
        for d in domains:
            d_root = os.path.join(split_root, d)
            for cname in os.listdir(d_root):
                cdir = os.path.join(d_root, cname)
                if not os.path.isdir(cdir):
                    continue
                if cname not in self.class_to_idx:
                    continue
                cls_idx = self.class_to_idx[cname]

                for fname in os.listdir(cdir):
                    fpath = os.path.join(cdir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    samples.append((fpath, cls_idx))

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class CORe50CIL(Dataset):
    """
    把 CORe50 的 train/test 统一成：
    - label = 物体 ID (o1..o50 -> 0..49)
    - train: 从 train/s*/o*/ 递归采样
    - test:  从 test/o*/      采样
    """
    def __init__(self, root, train=True, transform=None):
        """
        root: 指向 core50_128x128 的上一级目录，或者直接指向 core50_128x128
        """
        # 兼容两种传法：/path/to/core50_128x128 或 /path/to/
        if os.path.basename(root) != "core50_128x128":
            root = os.path.join(root, "core50_128x128")

        self.root = root
        self.train = train
        self.transform = transform

        split = "train" if train else "test"
        split_root = os.path.join(root, split)

        samples = []          # (img_path, obj_name, session_name)
        object_names_set = set()

        if train:
            # 结构：train/s*/o*/img
            for session in sorted(os.listdir(split_root)):
                session_dir = os.path.join(split_root, session)
                if not os.path.isdir(session_dir):
                    continue
                for obj in sorted(os.listdir(session_dir)):
                    obj_dir = os.path.join(session_dir, obj)
                    if not os.path.isdir(obj_dir):
                        continue
                    object_names_set.add(obj)
                    for img_name in sorted(os.listdir(obj_dir)):
                        img_path = os.path.join(obj_dir, img_name)
                        if not os.path.isfile(img_path):
                            continue
                        samples.append((img_path, obj, session))
        else:
            # 结构：test/o*/img
            for obj in sorted(os.listdir(split_root)):
                obj_dir = os.path.join(split_root, obj)
                if not os.path.isdir(obj_dir):
                    continue
                object_names_set.add(obj)
                for img_name in sorted(os.listdir(obj_dir)):
                    img_path = os.path.join(obj_dir, img_name)
                    if not os.path.isfile(img_path):
                        continue
                    # 用 "test" 作为一个虚拟的 session 名
                    samples.append((img_path, obj, "test"))

        # 统一的「物体列表」和 映射：o1..o50 -> 0..49
        self.classes = sorted(object_names_set)          # ['o1', ..., 'o50']
        self.class_to_idx = {obj: i for i, obj in enumerate(self.classes)}

        # 展开成真正要用的数据
        self.samples = [s[0] for s in samples]           # 所有图片路径
        self.targets = [self.class_to_idx[s[1]] for s in samples]  # 物体ID(0..49)
        self.domains = [s[2] for s in samples]           # session 名，后面想做 DIL 可以用

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        target = self.targets[index]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # 按你现在的 CIL 代码需求，返回 (img, target) 即可
        return img, target
