# **FUNGI**: **F**eatures from **UN**supervised **G**rad**I**ents

[Walter Simoncini](https://walter.ashita.nl/)<sup>1</sup>, [Andrei Bursuc](https://abursuc.github.io/)<sup>2</sup>, [Spyros Gidaris](https://scholar.google.fr/citations?user=7atfg7EAAAAJ&hl=en)<sup>2</sup>, [Yuki M. Asano](https://yukimasano.github.io/)<sup>1</sup>.

1. [QUVA Lab](https://ivi.fnwi.uva.nl/quva/), University of Amsterdam.
2. [valeo.ai](https://www.valeo.com/en/valeo-ai/), Paris, France.

This library implements our [No Train, all Gain: Self-Supervised Gradients Improve Deep Frozen Representations](https://fungi.ashita.nl/) paper. If you're looking for the code to replicate our experimental results please [click here](https://github.com/WalterSimoncini/no-train-all-gain).

The library allows you to extract **FUNGI**: **F**eatures from **UN**supervised **G**rad**I**ents from vision transformer backbones.
The FUNGI leverage the power of self-supervised losses to provide features that improve upon kNN-classification for images, text, audio and even semantic segmentation on images.

## Getting Started

You can install the `fungivision` package using the following command. The package requires `Python 3.10`.

```sh
pip install fungivision
```

We provide a quick demo of the library in `demo.ipynb`, where we extract FUNGI features for the [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset and a DINOv1 backbone. If you want to run the k-nearest neighbor classification evaluation make sure to also install `scikit-learn`!

## Example Usage

We provide an easy to use `FUNGIWrapper` to extract gradient features from any transformer backbone. First, initialize a torch `dataset` that returns a `(PIL.Image, label)`. It's important that **NO** transformation is applied to the raw images, as each SSL objective must apply its own augmentation independently. Second, initialize a transformer encoder, e.g. you can initialize [DINO](https://arxiv.org/abs/2104.14294) ViT-B/16 as follows:

```python
model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
```

After that, you can wrap the model with `FUNGIWrapper`.

```python
import torch
import torch.nn as nn

from tqdm import tqdm
from fungivision.wrapper import FUNGIWrapper
from fungivision.config import KLConfig, DINOConfig, SimCLRConfig


# Run the code on GPU if possible, or fallback on the CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Wrap the model using the FUNGI feature extractor
fungi = FUNGIWrapper(
    model=model,
    # The target layer is a dot-separated path to a linear layer within the model. The path
    # used here points to the attention output projection of the last transformer block.
    target_layer="blocks.11.attn.proj",
    device=device,
    # Using fp16 is ~2x faster, and the downstream performance is not affected.
    use_fp16=True,
    # The list of objectives for which the wrapper will extract gradient features.
    # As we use three objectives, the output features will have 3 * E dimensions,
    # where E is the dimensionality of the model embeddings. You can reduce the
    # feature dimensionality via PCA to maintain an iso-storage/retrieval cost.
    extractor_configs=[
        KLConfig(),
        DINOConfig(),
        # You can configure the self-supervised objectives by passing arguments
        # to their configuration objects. See each config dataclass in
        # src/fungivision/config for more details.
        SimCLRConfig(num_patches=4, stride_scale=6)
    ]
)

# You must call setup before extracting FUNGI features, as some objectives may
# require supporting data to compute the loss, e.g. the SimCLR negative batch
fungi.setup(dataset=train_dataset)
```

Once wrapper, you're ready to extract the gradient features!

```python
# Change as appropriate depending on your system
batch_size = 32
num_workers = 18
features = []

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    # This makes sure each iteration returns a list of images and a list of targets,
    # without the data loader creating a batch by itself, which may result in errors
    # as images may have a different size
    collate_fn=lambda batch: zip(*batch)
)

for images, _ in tqdm(data_loader):
    # The sub-components of each feature are already L2-normalized independently
    features.append(wrapper(images).cpu().float())
```

By default the wrapper does not extract the model embeddings, as each model requires its own inference transform. Assuming you've extracted them on your own, you can combine them with the gradient features as follows:

```python
embeddings = ...
embeddings = nn.functional.normalize(embeddings, dim=-1, p=2)

# Features are now [embeddings, KL gradients, DINO gradients, SimCLR gradients]
features = torch.cat([
    embeddings,
    features
], dim=-1)
```

### Creating your own SSL objective

You can create your own gradient-feature extractor, and to do so you just need to write two classes: a subclass of `BaseGradientExtractor` and its configuration dataclasss. Assuming your loss works with a single view, such as our KL objective, you just need to implement two methods:

```python
import torch
import torchvision.transforms.v2 as tf

from torch.nn.functional import log_softmax, softmax, kl_div
from fungivision.gradients.base_extractor import BaseGradientExtractor


class CustomGradientsExtractor(BaseGradientExtractor):
    def input_transform(self, input_dim: int) -> nn.Module:
        # Implement the data augmentation to be applied to each input image.
        # input_dim indicates the input dimensionality of the backbone.
        return tf.Compose([...])

    def compute_loss(self, latents: torch.Tensor, views_per_sample: int, **kwargs) -> torch.Tensor:
        # Given a batch of latent representations, compute the per-sample loss. It's
        # extremely important that the computational graph for each individual input
        # image is independent from the others, except for a final average of the
        # individual losses. If this constraint is not respected the per-sample gradients
        # will be contaminated by other batch items, and you will experience significant
        # performance fluctuations as you change the batch size (up to 10-20-30%!).
        #
        # You can also test for this mistake by comparing the gradients of the same
        # input sample when you forward it by itself and in a batch of 2 inputs. If
        # the gradients are significantly different when you're testing on a CPU then
        # the two batch items are probably interacting.
        # 
        # NOTE: on a GPU device the gradients may be slighty different as you change
        # the batch size even if you've done everything correctly, as modern GPUs pick
        # the most appropriate algorithm automatically, even if you force their behavior
        # to be deterministic.
        #
        # latents is a [B * V, E] tensor, where B is the batch size and V the number
        # of views (i.e. views_per_sample). If your data augmentation generates multiple
        # views per image you can reshape them in [B, V, E] using the following code:
        #
        # batch_size = latents.shape[0] // views_per_sample
        # latents = latents.reshape(batch_size, views_per_sample, -1)

        # In this function we implement our KL loss. Notice that the computational
        # graph of batch items is only fused at the end via reduction = "mean"
        latent_dim = latents.shape[1]

        uniform = (torch.ones(latent_dim) / latent_dim).to(self.device)

        softmax_uniform = softmax(uniform / self.temperature, dim=0)
        softmax_uniform = softmax_uniform.unsqueeze(dim=0).repeat(latents.shape[0], 1)

        softmax_latents = log_softmax(latents / self.temperature, dim=1)

        # NOTE: Always use a mean reduction!
        return kl_div(softmax_latents, softmax_uniform, reduction="mean")
```

If you accept custom configuration parameters, e.g. `self.temperature` in this case, you should also override the `__init__` method and add your parameters before the `**kwargs`. For more complex examples (that use multiple views per input image) see the DINO and SimCLR gradient extractors in `src/fungivision/gradients`. Once you've created your gradients extractor create a configuration dataclass as follows, which defines every user-customizable parameter for your extractor.

```python
from dataclasses import dataclass, asdict

from .extractor import CustomGradientsExtractor


@dataclass
class CustomConfig:
    temperature: float = 1

    def get_extractor(self, base_params: dict) -> CustomGradientsExtractor:
        # Create an instance of your feature extractor by merging the given
        # base parameters (which are common to all extractors) and your custom
        # parameters defined in this dataclass.
        params = base_params | asdict(self)

        return CustomGradientsExtractor(**params)
```

You can then use your gradients extractor with `FUNGIWrapper`!

```python
fungi = FUNGIWrapper(
    model=model,
    target_layer="blocks.11.attn.proj",
    device=device,
    use_fp16=True,
    extractor_configs=[
        CustomConfig(temperature=0.07)
    ]
)
```

## Related Repositories

The goal of this repository is providing an easy to use library for extracting FUNGI features from a vision transformer backbone. To reproduce the results shown in the paper please check out [this repository](https://github.com/WalterSimoncini/no-train-all-gain).

## Reference

If you found our work useful please cite us as follows:

```
@misc{simoncini2024fungi,
      title={No Train, all Gain: Self-Supervised Gradients Improve Deep Frozen Representations}, 
      author={Walter Simoncini and Spyros Gidaris and Andrei Bursuc and Yuki M. Asano},
      year={2024},
      eprint={2407.10964},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.10964}, 
}
```
