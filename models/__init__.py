from models.base import BaseVAE
from models.beta_vae import BetaVAE
from models.conditioned_vae import ConditionedVAE
from models.flow_vae import FlowVAE
from models.hierarchical_vae import HierarchicalVAE
from models.vanilla_vae import VanillaVAE

vae_models = {
    "VanillaVAE": VanillaVAE,
    "BetaVAE": BetaVAE,
    "FlowVAE": FlowVAE,
    "ConditionedVAE": ConditionedVAE,
    "HierarchicalVAE": HierarchicalVAE,
}


def getVAE(model_name: str, **kwargs) -> BaseVAE:
    if model_name in vae_models:
        model = vae_models[model_name](**kwargs)
    else:
        raise Exception(f"model {model_name} not found in library...")

    # verify BaseVAE
    assert isinstance(model, BaseVAE)

    return model


__all__ = ["getVAE", "BaseVAE"]
