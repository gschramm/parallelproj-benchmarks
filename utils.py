from __future__ import annotations

import array_api_compat.numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
from types import ModuleType


def gauss_smooth(x: npt.ArrayLike, sigma: float, xp: ModuleType,
                 dev: str) -> npt.ArrayLike:
    """array api compatible gaussian smoothing

    Parameters
    ----------
    x : npt.ArrayLike
        image array
    sigma : float
        sigma
    xp : ModuleType
        array module
    dev : str
        device (cpu or cuda)

    Returns
    -------
    npt.NDArrayLike
        smoothed image
    """

    if dev == 'cpu':
        return xp.asarray(gaussian_filter(np.asarray(x), sigma), device=dev)
    elif dev == 'cuda':
        import array_api_compat.cupy as cp
        import cupyx.scipy.ndimage as ndimagex
        return xp.asarray(ndimagex.gaussian_filter(cp.asarray(x), sigma),
                          device=dev)
    else:
        raise ValueError('Unsuported device')
