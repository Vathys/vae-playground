import argparse
import os

import torch
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import VAEDataset
from experiment import VAEExperiment
from utils import LogPerformanceCallback

parser = argparse.ArgumentParser(description="Generic runner for VAE models")
parser.add_argument(
    "--config",
    "-c",
    dest="filename",
    metavar="FILE",
    help="path to the config file",
    default="configs/vae.yaml",
)

args = parser.parse_args()
with open(args.filename, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger = TensorBoardLogger(
    save_dir=config["logging_params"]["save_dir"],
    name=config["model_params"]["name"],
)

# For reproducibility
seed_everything(config["experiment_params"]["manual_seed"], True)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

experiment = VAEExperiment(config["model_params"], config["experiment_params"])

data = VAEDataset(**config["data_params"])
data.setup()

runner = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=5,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val/loss",
            save_last=True,
            mode="min",
        ),
        LogPerformanceCallback(),
    ],
    max_epochs=config["experiment_params"]["max_epochs"],
    **config["trainer_params"],
)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
