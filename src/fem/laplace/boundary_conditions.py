from __future__ import annotations

import numpy as np

from fem.common.quadrature import line_quadrature


def polygon_orientation(points: np.ndarray) -> float:
    """
    Compute the signed area of a polygon to infer orientation.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Polygon vertices in order.

    Returns
    -------
    float
        Signed area (positive for CCW, negative for CW).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    x = points[:, 0]
    y = points[:, 1]
    area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return 0.5 * area2


def _quantize_points(points: np.ndarray, tol: float) -> np.ndarray:
    if tol <= 0:
        raise ValueError("tol must be positive")
    return np.round(points / tol).astype(np.int64)


def _normalize_boundary_points(boundary_points: np.ndarray, tol: float) -> np.ndarray:
    boundary_points = np.asarray(boundary_points, dtype=float)
    if boundary_points.shape[0] < 2:
        return boundary_points
    if np.allclose(boundary_points[0], boundary_points[-1], atol=tol, rtol=0.0):
        return boundary_points[:-1]
    return boundary_points


def map_boundary_points_to_nodes(
    points: np.ndarray,
    boundary_points: np.ndarray,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Map boundary vertices to node indices in the mesh by coordinate matching.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Mesh node coordinates.
    boundary_points : numpy.ndarray, shape (M, 2)
        Boundary coordinates in order.
    tol : float, optional
        Quantization tolerance for matching.

    Returns
    -------
    numpy.ndarray
        Array of node indices for each boundary point in order.
    """
    points = np.asarray(points, dtype=float)
    boundary_points = _normalize_boundary_points(boundary_points, tol)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if boundary_points.ndim != 2 or boundary_points.shape[1] != 2:
        raise ValueError("boundary_points must have shape (M, 2)")

    key = _quantize_points(points, tol)
    lookup: dict[tuple[int, int], int] = {}
    for idx, k in enumerate(map(tuple, key)):
        if k not in lookup:
            lookup[k] = idx

    boundary_key = _quantize_points(boundary_points, tol)
    indices = []
    missing = 0
    for k in map(tuple, boundary_key):
        idx = lookup.get(k)
        if idx is None:
            missing += 1
            indices.append(-1)
        else:
            indices.append(idx)

    if missing > 0:
        raise ValueError(f"Failed to map {missing} boundary points to mesh nodes.")

    return np.asarray(indices, dtype=np.int64)


def boundary_edges_from_points(
    points: np.ndarray,
    boundary_points: np.ndarray,
    *,
    tol: float = 1e-10,
    closed: bool = True,
) -> np.ndarray:
    """
    Build boundary edges as node index pairs from ordered boundary points.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Mesh node coordinates.
    boundary_points : numpy.ndarray, shape (M, 2)
        Ordered boundary coordinates.
    tol : float, optional
        Quantization tolerance for matching.
    closed : bool, optional
        If True, add edge from last point to first.

    Returns
    -------
    numpy.ndarray, shape (E, 2)
        Edge node indices in order.
    """
    boundary_points = _normalize_boundary_points(boundary_points, tol)
    boundary_nodes = map_boundary_points_to_nodes(points, boundary_points, tol=tol)
    if boundary_nodes.size < 2:
        raise ValueError("boundary_points must contain at least two points")

    start = boundary_nodes[:-1]
    end = boundary_nodes[1:]
    edges = np.column_stack([start, end])
    if closed:
        edges = np.vstack([edges, [boundary_nodes[-1], boundary_nodes[0]]])
    return edges.astype(np.int64)


def outward_edge_normals(
    boundary_points: np.ndarray,
    *,
    boundary_kind: str,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Compute outward unit normals for ordered boundary edges.

    Parameters
    ----------
    boundary_points : numpy.ndarray, shape (M, 2)
        Ordered boundary coordinates.
    boundary_kind : {"outer", "inner"}
        Whether this boundary encloses the domain or represents a hole.

    Returns
    -------
    numpy.ndarray, shape (E, 2)
        Outward unit normals per edge (same order as boundary segments).
    """
    boundary_points = _normalize_boundary_points(boundary_points, tol=tol)
    if boundary_points.ndim != 2 or boundary_points.shape[1] != 2:
        raise ValueError("boundary_points must have shape (M, 2)")
    if boundary_kind not in {"outer", "inner"}:
        raise ValueError("boundary_kind must be 'outer' or 'inner'")

    area = polygon_orientation(boundary_points)
    ccw = area > 0.0

    if boundary_kind == "outer":
        outward_is_right = ccw
    else:
        outward_is_right = not ccw

    p0 = boundary_points
    p1 = np.roll(boundary_points, -1, axis=0)
    edges = p1 - p0
    left = np.column_stack([-edges[:, 1], edges[:, 0]])
    right = np.column_stack([edges[:, 1], -edges[:, 0]])
    normals = right if outward_is_right else left
    norm = np.linalg.norm(normals, axis=1)
    eps = 1e-16
    normals /= np.maximum(norm[:, None], eps)
    return normals


def apply_dirichlet(
    stiffness,
    rhs: np.ndarray,
    nodes: np.ndarray,
    values: np.ndarray,
):
    """
    Apply Dirichlet boundary conditions by row/column elimination.

    Parameters
    ----------
    stiffness : scipy.sparse.spmatrix
        Global stiffness matrix.
    rhs : numpy.ndarray, shape (N,)
        Right-hand side vector.
    nodes : numpy.ndarray
        Indices of Dirichlet nodes.
    values : numpy.ndarray
        Dirichlet values at those nodes.

    Returns
    -------
    stiffness_bc : scipy.sparse.csr_matrix
        Modified stiffness matrix.
    rhs_bc : numpy.ndarray
        Modified right-hand side.
    """
    from scipy import sparse

    rhs = np.asarray(rhs, dtype=float).copy()
    nodes = np.asarray(nodes, dtype=np.int64)
    values = np.asarray(values, dtype=float)

    if nodes.size == 0:
        return stiffness.tocsr(), rhs
    if nodes.shape[0] != values.shape[0]:
        raise ValueError("nodes and values must have the same length")

    stiffness_csr = stiffness.tocsr()
    rhs -= stiffness_csr[:, nodes] @ values

    stiffness_lil = stiffness_csr.tolil()
    stiffness_lil[:, nodes] = 0.0
    for node in nodes:
        stiffness_lil.rows[node] = [node]
        stiffness_lil.data[node] = [1.0]
    rhs[nodes] = values

    return stiffness_lil.tocsr(), rhs


def apply_neumann(
    rhs: np.ndarray,
    points: np.ndarray,
    boundary_points: np.ndarray,
    flux,
    *,
    boundary_kind: str,
    tol: float = 1e-10,
    n_quad: int = 2,
) -> np.ndarray:
    """
    Add Neumann boundary contributions to the RHS.

    Parameters
    ----------
    rhs : numpy.ndarray, shape (N,)
        Right-hand side vector to update.
    points : numpy.ndarray, shape (N, 2)
        Mesh node coordinates.
    boundary_points : numpy.ndarray, shape (M, 2)
        Ordered boundary coordinates.
    flux : callable
        Function g(x, y, nx, ny) defining flux (dphi/dn).
    boundary_kind : {"outer", "inner"}
        Boundary type (outer domain or inner hole).
    tol : float, optional
        Quantization tolerance for boundary mapping.
    n_quad : int, optional
        Number of quadrature points per edge.

    Returns
    -------
    numpy.ndarray
        Updated RHS vector.
    """
    if flux is None:
        return rhs

    rhs = np.asarray(rhs, dtype=float).copy()
    points = np.asarray(points, dtype=float)
    boundary_points = _normalize_boundary_points(boundary_points, tol)

    edges = boundary_edges_from_points(points, boundary_points, tol=tol, closed=True)
    normals = outward_edge_normals(boundary_points, boundary_kind=boundary_kind, tol=tol)
    quad_s, quad_w = line_quadrature(n_quad)

    p0 = boundary_points
    p1 = np.roll(boundary_points, -1, axis=0)
    edge_vec = p1 - p0
    edge_len = np.linalg.norm(edge_vec, axis=1)

    for edge_idx, (i, j) in enumerate(edges):
        nvec = normals[edge_idx]
        if edge_len[edge_idx] <= 0.0:
            continue

        for s, w in zip(quad_s, quad_w):
            x = (1.0 - s) * p0[edge_idx, 0] + s * p1[edge_idx, 0]
            y = (1.0 - s) * p0[edge_idx, 1] + s * p1[edge_idx, 1]

            g_val = flux(x, y, nvec[0], nvec[1])
            if np.ndim(g_val) != 0:
                g_val = float(np.asarray(g_val).ravel()[0])

            n0 = 1.0 - s
            n1 = s
            weight = w * edge_len[edge_idx]
            rhs[i] += g_val * n0 * weight
            rhs[j] += g_val * n1 * weight

    return rhs
