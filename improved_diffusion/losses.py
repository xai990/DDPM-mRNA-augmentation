""" 
Helpers for various likelihood-based losses. These are ported from the original 
Ho et al. diffusion models codebase:
http://github.com/hojonathanho/diffusion
"""

import numpy as np 
import torch as th 

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to 
    scalars, among other use cases.
    """

    tensor = None 
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to 
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5* ( 
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the 
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi ) * (x + 0.044715 * th.pow(x,3))))


# the original code is based on image sample and the range is [-1, 1]

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a 
    given samples.

    :param x: the target sample. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1]. 
    :param means: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """

    assert x.shape == means.shape == log_scales.shape
    centered_x = x- means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0/255)