import os
import random
from pathlib import Path

DATA_PATH = Path("/mnt/ext-data-01/data")
TRAIN_NUM = 24000
VAL_NUM = 3000
TEST_NUM = 3000

fnames = []

for fname in os.listdir(DATA_PATH / "CelebAMask-HQ" / "CelebA-HQ-img"):
    fnames.append(fname)

print(f"Found {len(fnames)} files...")

test_set = random.sample(fnames, TEST_NUM)

rest = list(set(fnames) - set(test_set))

val_set = random.sample(rest, TEST_NUM)

train_set = list(set(rest) - set(val_set))

print(
    f"Splitting into {TEST_NUM} test samples, {VAL_NUM} validation samples and {TRAIN_NUM} train samples..."
)

with open(DATA_PATH / "CelebAMask-HQ" / "test.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(test_set))

with open(DATA_PATH / "CelebAMask-HQ" / "val.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(val_set))

with open(DATA_PATH / "CelebAMask-HQ" / "train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_set))
