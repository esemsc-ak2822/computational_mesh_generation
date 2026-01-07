from __future__ import annotations

import numpy as np


def _barycentric_weights(tri_pts: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates for a point in a triangle.
    """
    ax, ay = tri_pts[0]
    bx, by = tri_pts[1]
    cx, cy = tri_pts[2]
    px, py = point

    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-16:
        return np.array([-1.0, -1.0, -1.0])

    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
    w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
    w2 = 1.0 - w0 - w1
    return np.array([w0, w1, w2])


def interpolate_to_points(
    points: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    query_points: np.ndarray,
    *,
    fill_value: float = np.nan,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Interpolate nodal values to arbitrary points using barycentric coordinates.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Mesh node coordinates.
    triangles : numpy.ndarray, shape (T, 3)
        Triangle node indices.
    values : numpy.ndarray, shape (N,)
        Nodal values.
    query_points : numpy.ndarray, shape (Q, 2)
        Points to evaluate.
    fill_value : float, optional
        Value to use when a point is outside the mesh.
    tol : float, optional
        Tolerance for point-in-triangle checks.

    Returns
    -------
    numpy.ndarray
        Interpolated values at query points.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    query_points = np.asarray(query_points, dtype=float)

    if query_points.ndim != 2 or query_points.shape[1] != 2:
        raise ValueError("query_points must have shape (Q, 2)")

    tri_pts = points[triangles]
    tri_min = tri_pts.min(axis=1)
    tri_max = tri_pts.max(axis=1)

    out = np.full(query_points.shape[0], fill_value, dtype=float)
    for qi, qp in enumerate(query_points):
        candidates = np.where(
            (qp[0] >= tri_min[:, 0] - tol)
            & (qp[0] <= tri_max[:, 0] + tol)
            & (qp[1] >= tri_min[:, 1] - tol)
            & (qp[1] <= tri_max[:, 1] + tol)
        )[0]
        for ti in candidates:
            weights = _barycentric_weights(tri_pts[ti], qp)
            if np.all(weights >= -tol):
                out[qi] = np.dot(weights, values[triangles[ti]])
                break

    return out


class LinearTriInterpolator:
    """
    Reusable linear interpolator for a fixed mesh and nodal field.
    """

    def __init__(
        self,
        points: np.ndarray,
        triangles: np.ndarray,
        values: np.ndarray,
        *,
        tol: float = 1e-12,
    ) -> None:
        self.points = np.asarray(points, dtype=float)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.values = np.asarray(values, dtype=float)
        self.tol = tol

    def __call__(self, query_points: np.ndarray, *, fill_value: float = np.nan) -> np.ndarray:
        """
        Interpolate to query points.
        """
        return interpolate_to_points(
            self.points,
            self.triangles,
            self.values,
            query_points,
            fill_value=fill_value,
            tol=self.tol,
        )


def plot_scalar_field(
    points: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    *,
    ax=None,
    levels: int = 50,
    cmap: str = "viridis",
    show_mesh: bool = False,
    colorbar: bool = True,
):
    """
    Plot a scalar field over a triangular mesh using Matplotlib.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Mesh node coordinates.
    triangles : numpy.ndarray, shape (T, 3)
        Triangle node indices.
    values : numpy.ndarray, shape (N,)
        Nodal scalar values to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; if None, a new figure is created.
    levels : int, optional
        Number of contour levels for tricontourf.
    cmap : str, optional
        Matplotlib colormap name.
    show_mesh : bool, optional
        If True, overlay triangle edges.
    colorbar : bool, optional
        If True, add a colorbar to the figure.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot.
    """
    import matplotlib.pyplot as plt

    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    values = np.asarray(values, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    contour = ax.tricontourf(points[:, 0], points[:, 1], triangles, values, levels=levels, cmap=cmap)
    if show_mesh:
        ax.triplot(points[:, 0], points[:, 1], triangles, color="k", linewidth=0.3, alpha=0.4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if colorbar:
        plt.colorbar(contour, ax=ax, label="phi")
    return ax


def write_vtk_field(
    points: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    path,
    *,
    field_name: str = "phi",
):
    """
    Export a scalar field on a triangular mesh to VTK using meshio.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        Mesh node coordinates.
    triangles : numpy.ndarray, shape (T, 3)
        Triangle node indices.
    values : numpy.ndarray, shape (N,)
        Nodal scalar values to export.
    path : str or pathlib.Path
        Output VTK file path.
    field_name : str, optional
        Name for the point data field in the VTK file.
    """
    try:
        import meshio
    except ImportError as exc:
        raise ImportError("meshio is required for VTK export") from exc

    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    values = np.asarray(values, dtype=float)

    mesh = meshio.Mesh(points, [("triangle", triangles)], point_data={field_name: values})
    meshio.write(path, mesh)
