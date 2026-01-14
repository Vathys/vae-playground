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
    name=config["logging_params"]["name"],
)

# For reproducibility
global_seed = seed_everything(config["experiment_params"]["manual_seed"], False)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ckpt = config["trainer_params"].get("ckpt", None)
cont_run = config["trainer_params"].get("continue", False)
replace_config = config["trainer_params"].get("replace_config", True)

if ckpt is not None:
    if not replace_config:
        experiment = VAEExperiment.load_from_checkpoint(ckpt, map_location="cpu")
    else:
        experiment = VAEExperiment.load_from_checkpoint(
            ckpt,
            map_location="cpu",
            model_params=config["model_params"],
            experiment_params=config["experiment_params"],
        )
else:
    experiment = VAEExperiment(config["model_params"], config["experiment_params"])

data = VAEDataset(**config["data_params"], seed=global_seed)
data.setup()

# Trainer doesn't like extra variables
if "ckpt" in config["trainer_params"]:
    del config["trainer_params"]["ckpt"]
if "continue" in config["trainer_params"]:
    del config["trainer_params"]["continue"]
if "replace_config" in config["trainer_params"]:
    del config["trainer_params"]["replace_config"]

runner = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=-1,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val/loss",
            save_last=True,
            mode="min",
            every_n_epochs=5,
        ),
        LogPerformanceCallback(),
    ],
    **config["trainer_params"],
)

print(f"======= Training {config['model_params']['name']} =======")
if ckpt is not None and cont_run:
    runner.fit(experiment, datamodule=data, ckpt_path=ckpt)
else:
    runner.fit(experiment, datamodule=data)
