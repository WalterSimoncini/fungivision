import torch
import functools
import torch.nn as nn

from typing import List


def rsetattr(obj, attr, val):
    """
        Sets a nested attribute, e.g. model.encoder. Code based on:
        
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """
    pre, _, post = attr.rpartition('.')

    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
        Gets a nested attribute, e.g. model.encoder. Code based on:
        
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def model_feature_dim(model: nn.Module, device: torch.device, image_size: int = 224) -> int:
    """Given a feature extractor model returns the dimensionality of its features"""
    # Forward a random image through the model to retrieve a feature
    feature = model(torch.randn(1, 3, image_size, image_size).to(device))

    # Return the feature dimensionality
    return feature.squeeze().shape[0]


def freeze_model(model: nn.Module, exclusions: List[str] = []) -> None:
    """
        Freezes a model, excluding the weight and bias of the
        layer paths in the exclusions array
    """
    for param in model.parameters():
        param.requires_grad_(False)

    for layer_path in exclusions:
        layer = rgetattr(model, layer_path)
        layer.weight.requires_grad = True

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.requires_grad = True
