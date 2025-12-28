import numpy as np


def delaunay_triangulate(points: np.ndarray) -> np.ndarray:
    """
    Compute an unconstrained 2D Delaunay triangulation.

    Parameters
    ----------
    points : (N, 2) float array
        Node coordinates.

    Returns
    -------
    triangles : (T, 3) int array
        Triangle vertex indices into `points`.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points")

    try:
        from scipy.spatial import Delaunay
    except ImportError as e:
        raise ImportError("scipy is required (pip install scipy)") from e

    tri = Delaunay(points)
    return np.asarray(tri.simplices, dtype=np.int64)


def triangle_centroids(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Compute centroids for each triangle.

    Parameters
    ----------
    points : (N, 2) float array
    triangles : (T, 3) int array

    Returns
    -------
    centroids : (T, 2) float array
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")

    tri_pts = points[triangles]
    return tri_pts.mean(axis=1)


def filter_triangles_by_domain(
    points: np.ndarray,
    triangles: np.ndarray,
    outer_boundary: np.ndarray,
    airfoil_boundary: np.ndarray,
) -> np.ndarray:
    """
    Filter triangles by centroid inclusion: inside outer boundary and outside airfoil.

    Parameters
    ----------
    points : (N, 2) float array
    triangles : (T, 3) int array
    outer_boundary : (No, 2) float array
    airfoil_boundary : (Na, 2) float array

    Returns
    -------
    triangles_kept : (K, 3) int array
    """
    from meshgen.domain import points_in_polygon_mask

    centroids = triangle_centroids(points, triangles)
    inside_outer = points_in_polygon_mask(outer_boundary, centroids)
    inside_airfoil = points_in_polygon_mask(airfoil_boundary, centroids)

    keep = inside_outer & (~inside_airfoil)
    return triangles[keep]


def generate_mesh(
    points: np.ndarray,
    outer_boundary: np.ndarray,
    airfoil_boundary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience pipeline: Delaunay triangulation followed by domain filtering.

    Parameters
    ----------
    points : (N, 2) float array
    outer_boundary : (No, 2) float array
    airfoil_boundary : (Na, 2) float array

    Returns
    -------
    points : (N, 2) float array
    triangles_domain : (K, 3) int array
    """
    triangles = delaunay_triangulate(points)
    triangles_domain = filter_triangles_by_domain(points, triangles, outer_boundary, airfoil_boundary)
    return points, triangles_domain
