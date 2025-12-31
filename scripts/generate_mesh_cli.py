"""
Interactive/CLI mesh generator for NACA 4-digit airfoils.

Outputs:
- <prefix>.npz with points, triangles, airfoil_boundary, outer_boundary
- <prefix>.vtk (via meshio) for visualization/solver import
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from meshgen.domain import (
    airfoil_offset_rings,
    build_point_cloud,
    gen_outer_boundary,
    make_offset_distances,
    sample_farfield_points,
)
from meshgen.naca4 import gen_naca4
from meshgen.triangulate import generate_mesh

T = TypeVar("T")


def _prompt(value: T | None, text: str, default: T, cast: Callable[[str], T]) -> T:
    """
    Prompt for a value if not provided. Empty input -> default.
    """
    if value is not None:
        return value
    raw = input(f"{text} [{default}]: ").strip()
    if raw == "":
        return default
    return cast(raw)


def build_mesh(
    *,
    code: str,
    chord: float,
    n_chord: int,
    outer_radius_factor: float,
    n_outer: int,
    n_far: int,
    base_frac: float,
    n_layers: int,
    growth: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build mesh inputs: points, triangles, airfoil boundary, outer boundary.
    """
    rng = np.random.default_rng(seed)

    airfoil = gen_naca4(code, chord=chord, n_chord=n_chord)
    # Outer circle centered near mid-chord
    outer = gen_outer_boundary(radius=outer_radius_factor * chord, n_points=n_outer, center=(0.5 * chord, 0.0))

    distances = make_offset_distances(base=base_frac * chord, n_layers=n_layers, growth=growth)
    near = airfoil_offset_rings(airfoil, distances=distances)

    far = sample_farfield_points(
        outer_boundary=outer,
        airfoil_boundary=airfoil,
        n_points=n_far,
        min_dist_to_airfoil=distances[-1],
        rng=rng,
    )

    points = build_point_cloud(
        airfoil_boundary=airfoil,
        outer_boundary=outer,
        nearfield_points=near,
        farfield_points=far,
        deduplicate=True,
        dedup_tol=1e-12,
    )

    points, triangles = generate_mesh(points, outer_boundary=outer, airfoil_boundary=airfoil)
    return points, triangles, airfoil, outer


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive NACA airfoil mesh generator.")
    parser.add_argument("--code", type=str, help="NACA 4-digit code (e.g., 2412)")
    parser.add_argument("--chord", type=float, help="Chord length")
    parser.add_argument("--n-chord", type=int, help="Points along airfoil chord")
    parser.add_argument("--outer-radius-factor", type=float, help="Outer radius multiplier of chord")
    parser.add_argument("--n-outer", type=int, help="Points along outer boundary")
    parser.add_argument("--n-far", type=int, help="Farfield sample count")
    parser.add_argument("--base-frac", type=float, help="Base offset (fraction of chord) for nearfield")
    parser.add_argument("--n-layers", type=int, help="Number of nearfield layers")
    parser.add_argument("--growth", type=float, help="Growth factor between nearfield layers")
    parser.add_argument("--seed", type=int, help="RNG seed")
    parser.add_argument("--out-prefix", type=Path, default=Path("mesh_output"), help="Output prefix (no extension)")
    parser.add_argument("--no-vtk", action="store_true", help="Skip writing VTK file")
    args = parser.parse_args()

    # Interactive prompts when args omitted
    code = _prompt(args.code, "NACA code", "2412", str)
    chord = _prompt(args.chord, "Chord length", 1.0, float)
    n_chord = _prompt(args.n_chord, "Points on airfoil chord", 200, int)
    outer_radius_factor = _prompt(args.outer_radius_factor, "Outer radius factor (x chord)", 10.0, float)
    n_outer = _prompt(args.n_outer, "Points on outer boundary", 250, int)
    n_far = _prompt(args.n_far, "Farfield sample count", 2500, int)
    base_frac = _prompt(args.base_frac, "Nearfield base offset (fraction of chord)", 0.005, float)
    n_layers = _prompt(args.n_layers, "Nearfield layers", 5, int)
    growth = _prompt(args.growth, "Nearfield growth factor", 2.0, float)
    seed = _prompt(args.seed, "Random seed", 0, int)

    print("Generating mesh...")
    points, triangles, airfoil, outer = build_mesh(
        code=code,
        chord=chord,
        n_chord=n_chord,
        outer_radius_factor=outer_radius_factor,
        n_outer=n_outer,
        n_far=n_far,
        base_frac=base_frac,
        n_layers=n_layers,
        growth=growth,
        seed=seed,
    )

    out_npz = args.out_prefix.with_suffix(".npz")
    np.savez(
        out_npz,
        points=points,
        triangles=triangles,
        airfoil_boundary=airfoil,
        outer_boundary=outer,
    )
    print(f"Saved arrays to {out_npz}")

    if not args.no_vtk:
        try:
            import meshio
        except ImportError:
            print("meshio not installed; skipping VTK export.")
        else:
            mesh = meshio.Mesh(points, [("triangle", triangles)])
            out_vtk = args.out_prefix.with_suffix(".vtk")
            meshio.write(out_vtk, mesh)
            print(f"Saved VTK mesh to {out_vtk}")

    print("Done.")


if __name__ == "__main__":
    main()
