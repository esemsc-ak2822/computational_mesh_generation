import numpy as np


def triangle_area(coords: np.ndarray) -> float:
    """
    Compute the area of a triangle.

    Parameters
    ----------
    coords : numpy.ndarray, shape (3, 2)
        Triangle vertex coordinates (x, y).

    Returns
    -------
    float
        Triangle area (positive).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (3, 2):
        raise ValueError("coords must have shape (3, 2)")

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    return 0.5 * abs(det)


def shape_function_gradients(coords: np.ndarray) -> np.ndarray:
    """
    Compute gradients of P1 (linear) shape functions on a triangle.

    Parameters
    ----------
    coords : numpy.ndarray, shape (3, 2)
        Triangle vertex coordinates (x, y).

    Returns
    -------
    numpy.ndarray, shape (3, 2)
        Gradients of the three shape functions, one row per node.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (3, 2):
        raise ValueError("coords must have shape (3, 2)")

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(det) < 1e-16:
        raise ValueError("Degenerate triangle with near-zero area")

    grads = np.array(
        [
            [y2 - y3, x3 - x2],
            [y3 - y1, x1 - x3],
            [y1 - y2, x2 - x1],
        ],
        dtype=float,
    )
    grads /= det
    return grads
