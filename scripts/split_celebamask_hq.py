from pathlib import Path
import os
import random

DATA_PATH = Path("/mnt/ext-data-01/data")
SPLIT_RATIO = 0.25

fnames = []

for fname in os.listdir(DATA_PATH / "CelebAMask-HQ" / "CelebA-HQ-img"):
    fnames.append(fname)

print(f"Found {len(fnames)} files...")

num_test = int(len(fnames) * SPLIT_RATIO)

test_set = random.sample(fnames, num_test)

train_set = list(set(fnames) - set(test_set))

print(
    f"Splitting into {len(test_set)} test samples and {len(train_set)} train samples..."
)

with open(DATA_PATH / "CelebAMask-HQ" / "test.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(test_set))

with open(DATA_PATH / "CelebAMask-HQ" / "train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_set))
