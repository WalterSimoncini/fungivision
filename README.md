# FUNGI Features

This library allows you to extract **FUNGI**: **F**eatures from **UN**supervised **G**rad**I**ents from vision transformer backbones.

## Getting Started

You can build and install the `fungivision` package using the following command

```sh
pip install -e .
```

The package requires `Python 3.10`, but in principle you should be able to run `fungivision` on `Python 3.9` and greater. Modify the version as needed in `pyproject.toml`.

If you want to run the demo script, make sure to install `scikit-learn` as well. The `demo.py` script shows how FUNGI can be extracted from a DINOv1 backbone, and evaluates them in k-nearest neighbor classification using the [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset.
