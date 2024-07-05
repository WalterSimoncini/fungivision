"""
    Implements the random projection dimensionality reduction technique described in

    @inproceedings{achlioptas2001database,
        title={Database-friendly random projections},
        author={Achlioptas, Dimitris},
        booktitle={Proceedings of the twentieth ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems},
        pages={274--281},
        year={2001}
    }
"""
import torch

from typing import Tuple


def suggested_scaling_factor(projection_dim: int) -> float:
    """
        Return the scaling factor for the projected matrix according to

        @article{roburin2022take,
            title={Take One Gram of Neural Features, Get Enhanced Group Robustness},
            author={Roburin, Simon and Corbi{\`e}re, Charles and Puy, Gilles and Thome, Nicolas and Aubry, Matthieu and Marlet, Renaud and P{\'e}rez, Patrick},
            journal={arXiv preprint arXiv:2208.12625},
            year={2022}
        }

        Args:
            projection_dim (int): the output dimension of the linear projection.

        Returns:
            float: the scaling factor.
    """
    return 1.0 / torch.sqrt(torch.tensor(projection_dim))


def generate_projection_matrix(dims: Tuple[int, int], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
        Generates a matrix filled with 1 or -1 with a 50% probability.

        Args:
            dims (int, int):  the dimensions of the projection matrix,
                given as (out_dim, in_dim).

        Returns:
            torch.Tensor: the projection matrix, on the given device
            and with a float32 data type.
    """
    projection = ((torch.rand(dims) - 0.5) > 0).to(torch.int8).to(device)
    projection[projection == 0] = -1

    return projection.to(torch.float32)
