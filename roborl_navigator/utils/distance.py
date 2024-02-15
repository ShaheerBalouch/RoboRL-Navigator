import math

import numpy as np
from .formulas import spherical_distance


def regular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


def distance(a: np.ndarray, b: np.ndarray, cr=False) -> np.ndarray:
    assert a.shape == b.shape
    if not cr:
        return np.linalg.norm(a - b, axis=-1)
    if len(a.shape) == 2:
        return np.array([custom_distance(a[i], b[i]) for i in range(a.shape[0])])
    return custom_distance(a, b)


def custom_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # max: 1.24 mean 0.2
    pd = np.linalg.norm(a[:3] - b[:3])  # position_distance
    # Orientation: max distance = 6.38, mean = 1.52
    od = spherical_distance(a[3:], b[3:]) * 2
    return pd * 0.6 + od * 0.35 + pd * od


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dist
