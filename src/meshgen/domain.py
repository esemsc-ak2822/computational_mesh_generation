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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    boundary = gen_outer_boundary(radius=2.0, n_points=120, center=(0.5, -0.2))

    x, y = boundary[:, 0], boundary[:, 1]

    plt.figure(figsize=(4, 4))
    plt.plot(x, y, "-k", lw=1)
    plt.scatter(x[0], y[0], color="red", label="Start point")
    plt.axis("equal")
    plt.title("Outer Boundary Sanity Check")
    plt.legend()
    plt.grid(True)
    plt.show()
