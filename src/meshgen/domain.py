import numpy as np


def gen_outer_boundary(radius: float, n_points: int, center=(0.0, 0.0)):
    """
    Generate a circular outer boundary around a specified center.

    Parameters
    ----------
    radius : float
        Radius of the outer boundary.
    n_points : int
        Number of points uniformly distributed along the boundary.
    center : tuple[float, float], optional
        (x, y) coordinates of the circle center. Defaults to (0.0, 0.0).

    Returns
    -------
    numpy.ndarray
        Array of shape (n_points, 2) with points ordered counter-clockwise,
        starting on the positive x-axis relative to the center.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")

    cx, cy = center
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)

    return np.column_stack([x, y])

def points_in_polygon_mask(poly_pts: np.ndarray, query_pts: np.ndarray):
    poly_pts = np.asarray(poly_pts, dtype = float)
    query_pts = np.asarray(query_pts, dtype = float)

    try:
        from shapely.geometry import Point, Polygon
    except ImportError as e:
        raise ImportError(
            "Shapely is required for points_in_polygon_mask. Install it with: pip install shapely"
        ) from e
    
    poly = Polygon(poly_pts)
    if not poly.is_valid:
        # Attempt a simple repair (e.g., duplicate TE point)
        repaired = poly.buffer(0)
        if repaired.geom_type == "MultiPolygon":
            repaired = max(repaired.geoms, key=lambda g: g.area)
        if not repaired.is_valid:
            raise ValueError("Polygon is invalid (may be self-intersecting or degenerate).")
        poly = repaired
    
    mask = np.array([poly.contains(Point(x,y)) for x,y in query_pts], dtype = bool)
    return mask

def sample_farfield_points(
    outer_boundary: np.ndarray,
    airfoil_boundary: np.ndarray,
    n_points: int,
    *,
    rng: np.random.Generator | None = None,
    oversample_factor: float = 5.0,
    max_iters: int = 50,
    min_dist_to_airfoil: float | None = None,
) -> np.ndarray:
    """
    Sample (approximately) n_points random points in the annular region:
    inside the outer boundary polygon AND outside the airfoil polygon.

    Parameters
    ----------
    outer_boundary : np.ndarray, shape (N, 2)
        Outer domain boundary vertices (e.g., circle points), ordered (CW or CCW).
    airfoil_boundary : np.ndarray, shape (M, 2)
        Airfoil boundary vertices, ordered (CW or CCW).
    n_points : int
        Target number of farfield interior points to return.
    rng : np.random.Generator, optional
        RNG for reproducibility.
    oversample_factor : float, optional
        How many candidate points to draw per iteration relative to remaining need.
        Higher -> fewer iterations, more wasted samples.
    max_iters : int, optional
        Maximum rejection-sampling iterations.
    min_dist_to_airfoil : float, optional
        If provided, rejects points closer than this distance to the airfoil boundary.
        Useful to keep farfield points from crowding the nearfield refinement region.

    Returns
    -------
    np.ndarray, shape (K, 2)
        Sampled points, where K <= n_points if acceptance is low.
    """
    outer_boundary = np.asarray(outer_boundary, dtype=float)
    airfoil_boundary = np.asarray(airfoil_boundary, dtype=float)

    if outer_boundary.ndim != 2 or outer_boundary.shape[1] != 2:
        raise ValueError("outer_boundary must have shape (N, 2)")
    if airfoil_boundary.ndim != 2 or airfoil_boundary.shape[1] != 2:
        raise ValueError("airfoil_boundary must have shape (M, 2)")
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    if rng is None:
        rng = np.random.default_rng()

    # Bounding box for candidate sampling
    xmin, ymin = outer_boundary.min(axis=0)
    xmax, ymax = outer_boundary.max(axis=0)

    collected: list[np.ndarray] = []
    remaining = n_points

    if min_dist_to_airfoil is not None:
        try:
            from shapely.geometry import Point, Polygon
        except ImportError as e:
            raise ImportError(
                "Shapely is required for min_dist_to_airfoil. Install it with: pip install shapely"
            ) from e
        airfoil_poly = Polygon(airfoil_boundary)
        if not airfoil_poly.is_valid:
            # Attempt to repair common issues (e.g., duplicate trailing-edge point)
            repaired = airfoil_poly.buffer(0)
            if repaired.geom_type == "MultiPolygon":
                # Pick the largest valid piece
                repaired = max(repaired.geoms, key=lambda g: g.area)
            if repaired.is_valid:
                airfoil_poly = repaired
                airfoil_boundary = np.asarray(airfoil_poly.exterior.coords[:-1], dtype=float)
            else:
                raise ValueError("Airfoil polygon is invalid (self-intersecting or degenerate).")

    for _ in range(max_iters):
        if remaining <= 0:
            break

        n_cand = int(np.ceil(oversample_factor * remaining))
        cand = np.empty((n_cand, 2), dtype=float)
        cand[:, 0] = rng.uniform(xmin, xmax, size=n_cand)
        cand[:, 1] = rng.uniform(ymin, ymax, size=n_cand)

        # Keep only points inside the outer boundary.
        inside_outer = points_in_polygon_mask(outer_boundary, cand)

        # Remove points inside the airfoil (hole).
        inside_airfoil = points_in_polygon_mask(airfoil_boundary, cand)

        keep = inside_outer & (~inside_airfoil)
        cand_kept = cand[keep]

        if min_dist_to_airfoil is not None and cand_kept.size > 0:
            # Distance to polygon boundary is computed via shapely Point.distance(Polygon).
            dist_ok = np.array(
                [Point(x, y).distance(airfoil_poly) >= float(min_dist_to_airfoil) for x, y in cand_kept],
                dtype=bool,
            )
            cand_kept = cand_kept[dist_ok]

        if cand_kept.size == 0:
            continue

        take = min(remaining, cand_kept.shape[0])
        collected.append(cand_kept[:take])
        remaining -= take

    if not collected:
        return np.empty((0, 2), dtype=float)

    return np.vstack(collected)

def airfoil_offset_rings(
    airfoil_boundary: np.ndarray,
    distances: list[float],
    *,
    outward_check: bool = True,
) -> np.ndarray:
    """
    Generate nearfield refinement points by offsetting the airfoil boundary outward
    along an approximate normal direction, for multiple offset distances.

    Parameters
    ----------
    airfoil_boundary : np.ndarray, shape (N, 2)
        Airfoil boundary points in order (preferably CCW, non-self-intersecting).
        Does not need to be explicitly closed (first point repeated at end).
    distances : list[float]
        List of offset distances (in the same units as airfoil coordinates, e.g. chord-scaled).
        Example: [0.005, 0.01, 0.02, 0.04]
    outward_check : bool, optional
        If True, flips normals that point inward using a centroid-based heuristic.

    Returns
    -------
    np.ndarray, shape (len(distances) * N, 2)
        Stacked ring points. For each distance, returns N points offset from the boundary.
    """
    airfoil_boundary = np.asarray(airfoil_boundary, dtype=float)
    if airfoil_boundary.ndim != 2 or airfoil_boundary.shape[1] != 2:
        raise ValueError("airfoil_boundary must have shape (N, 2)")
    if airfoil_boundary.shape[0] < 3:
        raise ValueError("airfoil_boundary must contain at least 3 points")
    if len(distances) == 0:
        return np.empty((0, 2), dtype=float)

    dists = np.asarray(distances, dtype=float)
    if np.any(dists <= 0):
        raise ValueError("All distances must be positive")

    pts = airfoil_boundary

    # If boundary is explicitly closed (last == first), drop the last to avoid duplicates
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    N = pts.shape[0]

    # Approximate tangent at each point using centered differences:
    # tangent_i ~ p_{i+1} - p_{i-1} (with wrap-around)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    tangents = next_pts - prev_pts

    # Right-hand normal (rotate tangent clockwise): (tx, ty) -> (ty, -tx)
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])

    # Normalize normals to unit length
    norm = np.linalg.norm(normals, axis=1)
    # Avoid divide-by-zero (should be rare unless duplicate points)
    eps = 1e-12
    normals = normals / np.maximum(norm[:, None], eps)

    # Ensure normals point outward (centroid heuristic).
    # For a closed curve, vectors from centroid to boundary generally point outward.
    if outward_check:
        centroid = pts.mean(axis=0)
        radial = pts - centroid
        # If dot(normal, radial) < 0, normal points inward; flip it.
        flip = np.einsum("ij,ij->i", normals, radial) < 0
        normals[flip] *= -1.0

    # Build rings: for each distance d, ring = pts + d * normals
    rings = []
    for d in dists:
        rings.append(pts + d * normals)

    return np.vstack(rings)


def make_offset_distances(
    base: float,
    n_layers: int,
    growth: float = 2.0,
):
    """
    Build a geometric progression of offset distances for nearfield rings.

    Parameters
    ----------
    base : float
        First offset distance (e.g., initial boundary-layer thickness).
    n_layers : int
        Number of layers to generate.
    growth : float, optional
        Multiplicative growth factor between successive layers.

    Returns
    -------
    numpy.ndarray
        Array of length n_layers with distances: base * growth**i.
    """
    return np.asarray([base * (growth ** i) for i in range(n_layers)])


def build_point_cloud(
    airfoil_boundary: np.ndarray,
    outer_boundary: np.ndarray,
    *,
    nearfield_points: np.ndarray | None = None,
    farfield_points: np.ndarray | None = None,
    deduplicate: bool = True,
    dedup_tol: float = 1e-10,
) -> np.ndarray:
    """
    Combine boundary and interior point sets into a single point cloud for triangulation.

    Parameters
    ----------
    airfoil_boundary : np.ndarray, shape (Na, 2)
        Airfoil boundary points.
    outer_boundary : np.ndarray, shape (No, 2)
        Outer domain boundary points.
    nearfield_points : np.ndarray, shape (Nn, 2), optional (keyword-only)
        Near-airfoil refinement points (e.g. offset rings).
    farfield_points : np.ndarray, shape (Nf, 2), optional (keyword-only)
        Farfield interior points (coarse fill).
    deduplicate : bool, optional
        If True, remove near-duplicate points using a tolerance-based grid hash.
    dedup_tol : float, optional
        Tolerance for deduplication (in coordinate units). Points closer than this
        are treated as duplicates.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Combined point cloud suitable for triangulation.
    """
    airfoil_boundary = np.asarray(airfoil_boundary, dtype=float)
    outer_boundary = np.asarray(outer_boundary, dtype=float)

    if airfoil_boundary.ndim != 2 or airfoil_boundary.shape[1] != 2:
        raise ValueError("airfoil_boundary must have shape (N, 2)")
    if outer_boundary.ndim != 2 or outer_boundary.shape[1] != 2:
        raise ValueError("outer_boundary must have shape (N, 2)")

    parts = [airfoil_boundary, outer_boundary]

    if nearfield_points is not None:
        nearfield_points = np.asarray(nearfield_points, dtype=float)
        if nearfield_points.ndim != 2 or nearfield_points.shape[1] != 2:
            raise ValueError("nearfield_points must have shape (N, 2)")
        parts.append(nearfield_points)

    if farfield_points is not None:
        farfield_points = np.asarray(farfield_points, dtype=float)
        if farfield_points.ndim != 2 or farfield_points.shape[1] != 2:
            raise ValueError("farfield_points must have shape (N, 2)")
        parts.append(farfield_points)

    points = np.vstack(parts)

    if not deduplicate:
        return points

    if dedup_tol <= 0:
        raise ValueError("dedup_tol must be positive")

    # Tolerance-based dedup:
    # Quantize to a grid of size dedup_tol, then unique on integer grid coords.
    # This is fast and robust for removing exact/near duplicates.
    key = np.round(points / dedup_tol).astype(np.int64)
    _, unique_idx = np.unique(key, axis=0, return_index=True)

    points_unique = points[np.sort(unique_idx)]
    return points_unique


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from meshgen.naca4 import gen_naca4

    rng = np.random.default_rng(seed=0)

    # Outer circular boundary and NACA airfoil
    outer_boundary = gen_outer_boundary(radius=2.0, n_points=240, center=(0.5, -0.2))
    airfoil_boundary = gen_naca4("2412", chord=1.0, n_chord=200)
    airfoil_boundary[:, 0] += 0.5  # translate to match outer boundary center
    airfoil_boundary[:, 1] += -0.1

    try:
        farfield_pts = sample_farfield_points(
            outer_boundary=outer_boundary,
            airfoil_boundary=airfoil_boundary,
            n_points=600,
            rng=rng,
            oversample_factor=4.0,
            max_iters=75,
        )
    except ImportError as exc:
        print(f"Install shapely to sample farfield points: {exc}")
        farfield_pts = np.empty((0, 2), dtype=float)

    plt.figure(figsize=(5, 5))
    plt.plot(outer_boundary[:, 0], outer_boundary[:, 1], "-k", lw=1.0, label="Outer boundary")
    plt.plot(airfoil_boundary[:, 0], airfoil_boundary[:, 1], "-r", lw=1.0, label="Airfoil boundary")

    if farfield_pts.size > 0:
        plt.scatter(
            farfield_pts[:, 0],
            farfield_pts[:, 1],
            s=10,
            color="tab:blue",
            alpha=0.7,
            label="Candidate points",
        )

    # Mark a reference start on the outer boundary for orientation
    plt.scatter(outer_boundary[0, 0], outer_boundary[0, 1], color="black", s=18, label="Outer start")
    plt.axis("equal")
    plt.title("Outer/Airfoil Boundaries with Candidate Points")
    plt.legend()
    plt.grid(True)
    plt.show()
