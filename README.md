# VAE Playground

## Installation

Download CelebA from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Download [pretrained magface](https://github.com/IrvingMeng/MagFace?tab=readme-ov-file#model-zoo) (IResNet18 is recommended for training).

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

NOTE: `requirements-dev` is only for formatting packages such as black, or flake8. Can be skipped.

## Train

Make sure to point to location of CelebA installation in config files. For Conditioned VAE, make sure to point to MagFace location in config file.

```
python run.py --config config/<model-to-run>
```

Logging is done with Tensorboard: `tensorboard --logdir logs/`
