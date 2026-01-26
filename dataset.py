from pathlib import Path
from typing import List, Optional, Sequence, Union

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F


class RandomResolutionCollate:
    def __init__(
        self,
        min_res: int,
        max_res: int,
        divisible_by: int,
        interpolation: Union[F.InterpolationMode, int] = F.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        scale_longest: bool = True,
        base_seed: int = 0,
        deterministic: bool = True,
    ):
        self.gen = torch.Generator()
        self.base_seed = base_seed
        self.deterministic = deterministic
        self.min_res = (min_res // divisible_by) * divisible_by
        self.max_res = (max_res // divisible_by) * divisible_by
        self.divisible_by = divisible_by
        self.interpolation = interpolation
        self.antialias = antialias
        self.scale_longest = scale_longest

    def __call__(self, batch):
        _, h, w = F.get_dimensions(batch[0]["input"])

        mult = torch.randint(
            self.min_res // self.divisible_by,
            (self.max_res // self.divisible_by) + 1,
            (1,),
            dtype=torch.int,
        ).item()
        res = mult * self.divisible_by

        if self.scale_longest:
            if h > w:
                size = [res, int(res * (w / h))]
            else:
                size = [int(res * (h / w)), res]
        else:
            if h > w:
                size = [int(res * (h / w)), res]
            else:
                size = [res, int(res * (w / h))]

        new_size = [(s // self.divisible_by) * self.divisible_by for s in size]
        new_size = [max(s, self.divisible_by) for s in new_size]

        for sample in batch:
            sample["input"] = F.resize(
                sample["input"],
                new_size,
                interpolation=self.interpolation,
                antialias=self.antialias,
            )

        return default_collate(batch)


class CelebADataset(Dataset):
    def __init__(self, data_path, transform=None, split: str = "train"):
        self.data_path = Path(data_path) / "celeba"
        self.dataset = []
        self.transform = transform
        self.split = split

        list_file = self.data_path / "list_eval_partition.txt"
        images_dir = self.data_path / "img_align_celeba"

        with open(list_file, "r", encoding="utf-8") as f:
            self.dataset = [
                (
                    images_dir / file.strip().split(" ")[0],
                    int(file.strip().split(" ")[1]),
                )
                for file in f.readlines()
            ]

        split_map = {"train": 0, "val": 1, "test": 2}

        if split in split_map.keys():
            self.dataset = [image for image, s in self.dataset if s == split_map[split]]
        else:
            raise RuntimeError(f"split {split} not recognised")

        self.num_images = len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        image = Image.open(img_path)

        return {"input": self.transform(image)}

    def __len__(self):
        return self.num_images


class CelebAMaskHQDataset(Dataset):
    def __init__(self, data_path, transform=None, split: str = "train"):
        self.data_path = Path(data_path) / "CelebAMask-HQ"
        self.dataset = []
        self.transform = transform
        self.split = split

        list_file = self.data_path / f"{split}.txt"
        images_dir = self.data_path / "CelebA-HQ-img"

        with open(list_file, "r", encoding="utf-8") as f:
            self.dataset = [images_dir / file.strip() for file in f.readlines()]

        self.num_images = len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        image = Image.open(img_path)

        return {"input": self.transform(image)}

    def __len__(self):
        return self.num_images


class VAEDataset(L.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        data_path: str,
        seed: int,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Optional[Union[int, Sequence[int]]] = (256, 256),
        min_patch_size: int = 64,
        max_patch_size: int = 512,
        divisible_by: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.seed = seed
        self.dataset = dataset
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.divisible_by = divisible_by
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        raw_train = [T.RandomHorizontalFlip()]
        raw_val = []

        if self.dataset == "celeba":
            raw_train.append(T.CenterCrop(178))
            raw_val.append(T.CenterCrop(178))

        if self.patch_size is not None:
            raw_train.append(T.Resize(self.patch_size))
            raw_val.append(T.Resize(self.patch_size))

        raw_train.extend(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        raw_val.extend(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

        self.train_transform = T.Compose(raw_train)
        self.val_transform = T.Compose(raw_val)

        if self.dataset == "celeba":
            self.train_dataset = CelebADataset(
                self.data_dir, split="train", transform=self.train_transform
            )

            self.val_dataset = CelebADataset(
                self.data_dir, split="val", transform=self.val_transform
            )

            self.test_dataset = CelebADataset(
                self.data_dir, split="test", transform=self.val_transform
            )
        elif self.dataset == "celebamask_hq":
            self.train_dataset = CelebAMaskHQDataset(
                self.data_dir, split="train", transform=self.train_transform
            )

            self.val_dataset = CelebAMaskHQDataset(
                self.data_dir, split="val", transform=self.val_transform
            )

            self.test_dataset = CelebAMaskHQDataset(
                self.data_dir, split="test", transform=self.val_transform
            )
        else:
            raise ValueError(f"Dataset {self.dataset} not recognised")

    def train_dataloader(self):
        def train_worker_init_fn(worker_id):
            if self.trainer is not None:
                current_epoch = self.trainer.current_epoch
            else:
                current_epoch = 0

            worker_seed = (self.seed + worker_id + current_epoch) % (2**32)
            torch.manual_seed(worker_seed)

        dl_kwargs = {
            "batch_size": self.train_batch_size,
            "num_workers": self.num_workers,
            "shuffle": True,
            "pin_memory": self.pin_memory,
            "worker_init_fn": train_worker_init_fn,
        }
        if self.patch_size is None:
            dl_kwargs["collate_fn"] = RandomResolutionCollate(
                self.min_patch_size, self.max_patch_size, self.divisible_by
            )
        return DataLoader(self.train_dataset, **dl_kwargs)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        def val_worker_init_fn(worker_id):
            worker_seed = (self.seed + worker_id) % (2**32)
            torch.manual_seed(worker_seed)

        dl_kwargs = {
            "batch_size": self.val_batch_size,
            "num_workers": self.num_workers,
            "shuffle": False,
            "pin_memory": self.pin_memory,
            "worker_init_fn": val_worker_init_fn,
        }
        if self.patch_size is None:
            dl_kwargs["collate_fn"] = RandomResolutionCollate(
                self.min_patch_size, self.max_patch_size, self.divisible_by
            )
        return DataLoader(self.val_dataset, **dl_kwargs)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        def test_worker_init_fn(worker_id):
            worker_seed = (self.seed + worker_id) % (2**32)
            torch.manual_seed(worker_seed)

        dl_kwargs = {
            "batch_size": self.val_batch_size,
            "num_workers": self.num_workers,
            "shuffle": True,
            "pin_memory": self.pin_memory,
            "worker_init_fn": test_worker_init_fn,
        }
        return DataLoader(self.test_dataset, **dl_kwargs)
