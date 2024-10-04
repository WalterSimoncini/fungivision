import torch
import functools
import torch.nn as nn

from typing import List, Any


def rsetattr(obj: Any, attr: str, val: Any):
    """
        Sets a nested attribute, e.g. model.encoder. Code based on:
        
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

        Args:
            obj (Any): an object to get a nested field from.
            attr (str): nested attribute, with dot-separated path components,
                e.g. blocks.11.attn.proj.
            value (Any): the value to which the nested attribute should be set.
    """
    pre, _, post = attr.rpartition('.')

    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj: Any, attr: str, *args) -> Any:
    """
        Gets a nested attribute, e.g. model.encoder. Code based on:
        
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

        Args:
            obj (Any): an object to get a nested field from.
            attr (str): nested attribute, with dot-separated path components,
                e.g. blocks.11.attn.proj.

        Returns:
            Any: the value of the field at the nested path
    """
    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def model_feature_dim(model: nn.Module, device: torch.device, image_size: int = 224) -> int:
    """
        Given a feature extractor model returns the dimensionality of its features.

        Args:
            model (nn.Module): a feature extractor model.
            device (torch.device): the device the model is on.
            image_size (int): the model input size

        Returns:
            int: the model embedding size
    """
    # Forward a random image through the model to retrieve a feature
    feature = model(torch.randn(1, 3, image_size, image_size).to(device))

    # Return the feature dimensionality
    return feature.squeeze().shape[0]


def freeze_model(model: nn.Module, exclusions: List[str] = []) -> None:
    """
        Freezes a model, excluding the weight and bias of the
        layer paths in the exclusions array.

        Args:
            model (nn.Module): a model.
            exclusions (List[str]): list of strings defining layer paths not to
                be frozen. Each path must be in the form blocks.11.attn.proj, i.e.
                dot-separated path components for nested fields.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    for layer_path in exclusions:
        layer = rgetattr(model, layer_path)
        layer.weight.requires_grad = True

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.requires_grad = True
