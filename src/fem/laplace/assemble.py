from __future__ import annotations

import numpy as np

from fem.common.basis_p1 import shape_function_gradients, triangle_area


def assemble_laplace_stiffness(points: np.ndarray, triangles: np.ndarray):
    """
    Assemble the global stiffness matrix for the Laplace operator.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Mesh node coordinates.
    triangles : numpy.ndarray, shape (T, 3)
        Triangle node indices.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse stiffness matrix.
    """
    from scipy import sparse

    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")

    n_nodes = points.shape[0]
    n_tri = triangles.shape[0]
    nnz = n_tri * 9

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=float)

    idx = 0
    for tri in triangles:
        coords = points[tri]
        grads = shape_function_gradients(coords)
        area = triangle_area(coords)
        local = area * (grads @ grads.T)

        for a in range(3):
            for b in range(3):
                rows[idx] = tri[a]
                cols[idx] = tri[b]
                data[idx] = local[a, b]
                idx += 1

    stiffness = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    return stiffness.tocsr()


def assemble_load(
    points: np.ndarray,
    triangles: np.ndarray,
    source=None,
) -> np.ndarray:
    """
    Assemble the load vector for a volumetric source term.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Mesh node coordinates.
    triangles : numpy.ndarray, shape (T, 3)
        Triangle node indices.
    source : callable, optional
        Function f(x, y) defining the source term. If None, returns zeros.

    Returns
    -------
    numpy.ndarray
        Load vector of length N.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    n_nodes = points.shape[0]

    rhs = np.zeros(n_nodes, dtype=float)
    if source is None:
        return rhs

    for tri in triangles:
        coords = points[tri]
        area = triangle_area(coords)
        centroid = coords.mean(axis=0)
        f_val = source(centroid[0], centroid[1])
        if np.ndim(f_val) != 0:
            f_val = float(np.asarray(f_val).ravel()[0])
        rhs[tri] += (f_val * area / 3.0)

    return rhs
