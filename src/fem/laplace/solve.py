from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from fem.laplace.assemble import assemble_laplace_stiffness, assemble_load
from fem.laplace.boundary_conditions import apply_dirichlet, apply_neumann, map_boundary_points_to_nodes


PotentialFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
FluxFn = Callable[[float, float, float, float], float]


@dataclass(frozen=True)
class MeshGeometry:
    """
    Container for mesh geometry needed by the FEM solver.

    Attributes
    ----------
    points : numpy.ndarray, shape (N, 2)
        Mesh node coordinates.
    triangles : numpy.ndarray, shape (T, 3)
        Triangle node indices.
    airfoil_boundary : numpy.ndarray, shape (Na, 2)
        Airfoil boundary coordinates (ordered).
    outer_boundary : numpy.ndarray, shape (No, 2)
        Outer boundary coordinates (ordered).
    """

    points: np.ndarray
    triangles: np.ndarray
    airfoil_boundary: np.ndarray
    outer_boundary: np.ndarray

    @classmethod
    def from_npz(cls, path) -> "MeshGeometry":
        """
        Load mesh geometry from a NumPy .npz produced by the mesh generator.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the .npz file.

        Returns
        -------
        MeshGeometry
            Loaded mesh geometry.
        """
        data = np.load(path)
        return cls(
            points=np.asarray(data["points"], dtype=float),
            triangles=np.asarray(data["triangles"], dtype=np.int64),
            airfoil_boundary=np.asarray(data["airfoil_boundary"], dtype=float),
            outer_boundary=np.asarray(data["outer_boundary"], dtype=float),
        )


def _evaluate_potential(fn: PotentialFn, coords: np.ndarray) -> np.ndarray:
    """
    Evaluate a farfield potential function on a list of coordinates.
    """
    try:
        values = fn(coords[:, 0], coords[:, 1])
    except TypeError:
        values = fn(coords)
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(coords.shape[0], float(values))
    elif values.ndim > 1:
        values = values.reshape(-1)
    if values.shape[0] != coords.shape[0]:
        raise ValueError("farfield_potential must return one value per coordinate")
    return values


def solve_laplace(
    farfield_potential: PotentialFn,
    geometry: MeshGeometry,
    *,
    airfoil_flux: FluxFn | None = None,
    source=None,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Solve the Laplace equation for potential flow around an airfoil.

    Parameters
    ----------
    farfield_potential : callable
        Function phi(x, y) defining Dirichlet values on the outer boundary.
    geometry : MeshGeometry
        Mesh geometry data.
    airfoil_flux : callable, optional
        Neumann flux g(x, y, nx, ny) on the airfoil boundary. If None, defaults
        to zero-flux (no-penetration).
    source : callable, optional
        Volumetric source term f(x, y). If None, assumes zero.
    tol : float, optional
        Coordinate tolerance for boundary mapping.

    Returns
    -------
    numpy.ndarray
        Nodal potential values phi at mesh nodes.
    """
    points = np.asarray(geometry.points, dtype=float)
    triangles = np.asarray(geometry.triangles, dtype=np.int64)

    stiffness = assemble_laplace_stiffness(points, triangles)
    rhs = assemble_load(points, triangles, source=source)

    if airfoil_flux is None:
        airfoil_flux = lambda x, y, nx, ny: 0.0

    rhs = apply_neumann(
        rhs,
        points,
        geometry.airfoil_boundary,
        airfoil_flux,
        boundary_kind="inner",
        tol=tol,
    )

    outer_nodes = map_boundary_points_to_nodes(points, geometry.outer_boundary, tol=tol)
    outer_values = _evaluate_potential(farfield_potential, points[outer_nodes])
    stiffness_bc, rhs_bc = apply_dirichlet(stiffness, rhs, outer_nodes, outer_values)

    from scipy.sparse.linalg import spsolve

    phi = spsolve(stiffness_bc, rhs_bc)
    return np.asarray(phi, dtype=float)


class LaplaceOperator:
    """
    Operator wrapper for DeepONet: (farfield potential, geometry) -> phi(x, y).
    """

    def __init__(
        self,
        geometry: MeshGeometry,
        *,
        airfoil_flux: FluxFn | None = None,
        source=None,
        tol: float = 1e-10,
    ) -> None:
        self.geometry = geometry
        self.airfoil_flux = airfoil_flux
        self.source = source
        self.tol = tol

    def __call__(self, farfield_potential: PotentialFn) -> np.ndarray:
        """
        Evaluate the FEM operator for a given farfield potential function.

        Parameters
        ----------
        farfield_potential : callable
            Function phi(x, y) defining Dirichlet values on the outer boundary.

        Returns
        -------
        numpy.ndarray
            Nodal potential values phi at mesh nodes.
        """
        return solve_laplace(
            farfield_potential,
            self.geometry,
            airfoil_flux=self.airfoil_flux,
            source=self.source,
            tol=self.tol,
        )


def fem_operator(
    farfield_potential: PotentialFn,
    geometry: MeshGeometry,
    *,
    airfoil_flux: FluxFn | None = None,
    source=None,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Functional interface matching the operator signature used for DeepONet.

    Parameters
    ----------
    farfield_potential : callable
        Function phi(x, y) defining Dirichlet values on the outer boundary.
    geometry : MeshGeometry
        Mesh geometry data.
    airfoil_flux : callable, optional
        Neumann flux g(x, y, nx, ny) on the airfoil boundary.
    source : callable, optional
        Volumetric source term f(x, y).
    tol : float, optional
        Coordinate tolerance for boundary mapping.

    Returns
    -------
    numpy.ndarray
        Nodal potential values phi at mesh nodes.
    """
    return solve_laplace(
        farfield_potential,
        geometry,
        airfoil_flux=airfoil_flux,
        source=source,
        tol=tol,
    )
