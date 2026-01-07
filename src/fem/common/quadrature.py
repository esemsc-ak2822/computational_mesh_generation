import numpy as np


def gauss_legendre_1d(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return Gauss-Legendre points and weights on [-1, 1].

    Parameters
    ----------
    n_points : int
        Number of quadrature points.

    Returns
    -------
    points : numpy.ndarray
        Quadrature points in [-1, 1].
    weights : numpy.ndarray
        Quadrature weights.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    points, weights = np.polynomial.legendre.leggauss(n_points)
    return points.astype(float), weights.astype(float)


def line_quadrature(n_points: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Return quadrature points and weights on the unit segment [0, 1].

    Parameters
    ----------
    n_points : int, optional
        Number of quadrature points on the segment.

    Returns
    -------
    points : numpy.ndarray
        Quadrature points in [0, 1].
    weights : numpy.ndarray
        Quadrature weights.
    """
    points, weights = gauss_legendre_1d(n_points)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    return points, weights
