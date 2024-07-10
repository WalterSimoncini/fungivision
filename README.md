# **FUNGI**: **F**eatures from **UN**supervised **G**rad**I**ents

Walter Simoncini<sup>1</sup>, Andrei Bursuc<sup>2</sup>, Spyros Gidaris<sup>2</sup>, Yuki M. Asano<sup>1</sup>.

1. [QUVA Lab](https://ivi.fnwi.uva.nl/quva/), University of Amsterdam.
2. [valeo.ai](https://www.valeo.com/en/valeo-ai/), Paris, France.

This is the code for our [paper name]() paper.

The library allows you to extract **FUNGI**: **F**eatures from **UN**supervised **G**rad**I**ents from vision transformer backbones.
The FUNGI leverage the power of self-supervised losses to provide features that improve upon kNN-classification for images, text, audio and even semantic segmentation on images.

## Getting Started

You can build and install the `fungivision` package using the following command

```sh
pip install -e .
```

The package requires `Python 3.10`, but in principle you should be able to run `fungivision` on `Python 3.9` and greater. Modify the version as needed in `pyproject.toml`.

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

Something something...

## Related Repositories

The goal of this repository is providing an easy to use library for extracting FUNGI features from a vision transformer backbone. To reproduce the results shown in the paper please use the following repositories:

- [Vision](): (TODO) k-nearest neighbor and linear classification of images and image retrieval experiments.
- [Text](https://github.com/WalterSimoncini/fungi-text): k-nearest neighbor text classification using FUNGI obtained from text encoders.
- [Audio](https://github.com/WalterSimoncini/fungi-ssast): k-nearest neighbor audio classification using FUNGI obtained from an SSAST backbone.
- [ICL/HummingBird](https://github.com/WalterSimoncini/fungi-hummingbird): retrieval-based vision in-context learning evaluation on semantic segmentation tasks.

## Reference

TODO: add bibtex
```
simoncini2024fungi
```
