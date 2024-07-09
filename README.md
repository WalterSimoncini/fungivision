# **FUNGI**: **F**eatures from **UN**supervised **G**rad**I**ents
TODO Authornames

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

## Example usage
```python
todo: example code
```

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
