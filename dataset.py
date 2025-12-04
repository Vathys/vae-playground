from typing import List, Optional, Sequence, Union
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image


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

        list_file = self.data_path / ("train.txt" if split == "train" else "test.txt")
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
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.dataset = dataset
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        raw_train = [T.RandomHorizontalFlip()]
        raw_val = []

        if self.dataset == "celeba":
            raw_train.append(T.CenterCrop(178))
            raw_val.append(T.CenterCrop(178))

        raw_train.extend(
            [
                T.Resize(self.patch_size),
                T.ToTensor(),
                T.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )
        raw_val.extend(
            [
                T.Resize(self.patch_size),
                T.ToTensor(),
                T.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )

        if self.dataset == "celeba":
            self.train_dataset = CelebADataset(
                self.data_dir, split="train", transform=T.Compose(raw_train)
            )

            self.val_dataset = CelebADataset(
                self.data_dir, split="test", transform=T.Compose(raw_val)
            )
        elif self.dataset == "celebamask_hq":
            self.train_dataset = CelebAMaskHQDataset(
                self.data_dir, split="train", transform=T.Compose(raw_train)
            )

            self.val_dataset = CelebAMaskHQDataset(
                self.data_dir, split="test", transform=T.Compose(raw_val)
            )
        else:
            raise ValueError(f"Dataset {self.dataset} not recognised")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
