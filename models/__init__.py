from models.base import *
from models.vanilla_vae import *
from models.beta_vae import *
from models.flow_vae import *

vae_models = {"VanillaVAE": VanillaVAE, "BetaVAE": BetaVAE, "FlowVAE": FlowVAE}
