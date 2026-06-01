import argparse
import hashlib
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fem.common.basis_p1 import shape_function_gradients
from fem.laplace.solve import MeshGeometry, solve_laplace
from meshgen.domain import airfoil_offset_rings, build_point_cloud, gen_outer_boundary, sample_farfield_points
from meshgen.naca4 import gen_naca4
from meshgen.triangulate import generate_mesh


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def run_simulation(naca_code: str, alpha_deg: float, cfg: dict, sim_seed: int):
    profiles: dict[str, float] = {}
    alpha_rad = np.radians(alpha_deg)

    chord = float(cfg["chord"])
    n_chord = int(cfg["n_chord"])
    outer_radius = float(cfg["outer_radius"])
    n_outer = int(cfg["n_outer"])

    airfoil_boundary = gen_naca4(naca_code, chord=chord, n_chord=n_chord)
    outer_boundary = gen_outer_boundary(radius=outer_radius, n_points=n_outer)

    t_start = time.perf_counter()
    rng = np.random.default_rng(seed=sim_seed)

    near_points = airfoil_offset_rings(
        airfoil_boundary,
        distances=[float(d) for d in cfg["mesh_nearfield_offsets"]],
    )

    farfield_pts = sample_farfield_points(
        outer_boundary=outer_boundary,
        airfoil_boundary=airfoil_boundary,
        n_points=int(cfg["mesh_farfield_points"]),
        min_dist_to_airfoil=float(cfg["mesh_min_farfield_dist"]),
        rng=rng,
    )

    points = build_point_cloud(
        airfoil_boundary=airfoil_boundary,
        outer_boundary=outer_boundary,
        nearfield_points=near_points,
        farfield_points=farfield_pts,
    )
    pts, triangles = generate_mesh(points, outer_boundary, airfoil_boundary)
    profiles["mesh_gen_sec"] = time.perf_counter() - t_start

    t_start = time.perf_counter()

    def farfield_phi(x: np.ndarray, y: np.ndarray):
        return float(cfg["v_inf"]) * (x * np.cos(alpha_rad) + y * np.sin(alpha_rad))

    geometry = MeshGeometry(
        points=pts,
        triangles=triangles,
        airfoil_boundary=airfoil_boundary,
        outer_boundary=outer_boundary,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        phi = solve_laplace(farfield_phi, geometry)
    profiles["fem_solve_sec"] = time.perf_counter() - t_start

    t_start = time.perf_counter()
    n_nodes = pts.shape[0]
    n_tri = triangles.shape[0]

    velocity_tri = np.zeros((n_tri, 2), dtype=float)
    for i, tri in enumerate(triangles):
        coords = pts[tri]
        grads = shape_function_gradients(coords)
        velocity_tri[i] = phi[tri] @ grads

    velocity_nodes = np.zeros((n_nodes, 2), dtype=float)
    counts = np.zeros(n_nodes, dtype=float)
    for i, tri in enumerate(triangles):
        velocity_nodes[tri] += velocity_tri[i]
        counts[tri] += 1.0
    if np.any(counts == 0.0):
        raise ValueError("Invalid mesh: at least one node is not connected to any triangle")
    velocity_nodes /= counts[:, None]

    v_mag_sq = np.sum(velocity_nodes**2, axis=1)
    pressure_nodes = 0.5 * (float(cfg["v_inf"]) ** 2 - v_mag_sq)

    if not np.isfinite(phi).all():
        raise ValueError("Non-finite potential field encountered")
    if not np.isfinite(velocity_nodes).all():
        raise ValueError("Non-finite velocity field encountered")
    if not np.isfinite(pressure_nodes).all():
        raise ValueError("Non-finite pressure field encountered")
    profiles["post_process_sec"] = time.perf_counter() - t_start

    return geometry, phi, velocity_nodes, pressure_nodes, profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FEM-only dataset (no SDF/BC sensor vectors).")
    parser.add_argument("--output", type=Path, default=Path("dataset_fem_v1.npz"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-airfoils", type=int, default=15)
    parser.add_argument("--num-alphas", type=int, default=15)
    parser.add_argument("--alpha-min", type=float, default=-8.0)
    parser.add_argument("--alpha-max", type=float, default=18.0)
    return parser.parse_args()


def main():
    args = parse_args()

    candidate_airfoils = [
        "0006",
        "0009",
        "0010",
        "0012",
        "0015",
        "1410",
        "2410",
        "2412",
        "2415",
        "2424",
        "4412",
        "4415",
        "6409",
        "6412",
        "7409",
        "7412",
        "8412",
        "9412",
    ]
    n_airfoils = min(max(1, int(args.num_airfoils)), len(candidate_airfoils))
    naca_airfoils = candidate_airfoils[:n_airfoils]
    alphas = np.linspace(float(args.alpha_min), float(args.alpha_max), int(args.num_alphas))

    cfg = {
        "dataset_version": "fem_only_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "seed": int(args.seed),
        "naca_airfoils": naca_airfoils,
        "alphas_deg": alphas.tolist(),
        "v_inf": 1.0,
        "chord": 1.0,
        "n_chord": 180,
        "outer_radius": 5.0,
        "n_outer": 180,
        "mesh_farfield_points": 2200,
        "mesh_nearfield_offsets": [0.005, 0.01, 0.02, 0.04],
        "mesh_min_farfield_dist": 0.04,
    }

    cfg_json = json.dumps(cfg, sort_keys=True)
    cfg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()[:12]
    cfg["config_hash"] = cfg_hash

    all_phi_targets = []
    all_velocity_targets = []
    all_pressure_targets = []
    all_mesh_coords = []
    all_mesh_triangles = []
    all_airfoil_boundaries = []
    all_outer_boundaries = []
    simulation_metadata = []

    total_sims = len(naca_airfoils) * len(alphas)
    print(f"Starting FEM data generation: {len(naca_airfoils)} airfoils x {len(alphas)} alphas = {total_sims} simulations")
    print(f"Config hash: {cfg_hash}")

    sim_counter = 0
    base_rng = np.random.default_rng(seed=cfg["seed"])

    for code in naca_airfoils:
        for alpha in alphas:
            sim_counter += 1
            sim_seed = int(base_rng.integers(0, 2**31 - 1))
            print(f"[{sim_counter:04d}/{total_sims}] NACA {code}, alpha={alpha:.3f} deg")

            try:
                geometry, phi, velocity, pressure, profiles = run_simulation(
                    code,
                    float(alpha),
                    cfg,
                    sim_seed=sim_seed,
                )

                all_phi_targets.append(phi)
                all_velocity_targets.append(velocity)
                all_pressure_targets.append(pressure)
                all_mesh_coords.append(geometry.points)
                all_mesh_triangles.append(geometry.triangles)
                all_airfoil_boundaries.append(geometry.airfoil_boundary)
                all_outer_boundaries.append(geometry.outer_boundary)

                meta = {
                    "sim_id": len(all_phi_targets) - 1,
                    "config_hash": cfg_hash,
                    "naca": code,
                    "alpha": float(alpha),
                    "v_inf": float(cfg["v_inf"]),
                    "sim_seed": sim_seed,
                    "n_nodes": int(len(phi)),
                    "n_triangles": int(len(geometry.triangles)),
                    "airfoil_boundary_points": int(geometry.airfoil_boundary.shape[0]),
                }
                meta.update(profiles)
                simulation_metadata.append(meta)
            except Exception as exc:
                print(f"Failed simulation for NACA {code} alpha={alpha:.3f}: {exc}")

    np.savez_compressed(
        args.output,
        phi_targets=np.array(all_phi_targets, dtype=object),
        velocity_targets=np.array(all_velocity_targets, dtype=object),
        pressure_targets=np.array(all_pressure_targets, dtype=object),
        mesh_coords=np.array(all_mesh_coords, dtype=object),
        mesh_triangles=np.array(all_mesh_triangles, dtype=object),
        airfoil_boundaries=np.array(all_airfoil_boundaries, dtype=object),
        outer_boundaries=np.array(all_outer_boundaries, dtype=object),
        metadata=np.array(simulation_metadata, dtype=object),
        fem_config_json=np.array(cfg_json),
        fem_config_hash=np.array(cfg_hash),
    )

    config_sidecar = args.output.with_suffix(".meta.json")
    config_sidecar.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    print(f"FEM dataset generated and saved to {args.output}")
    print(f"FEM config metadata saved to {config_sidecar}")
    print(f"num_valid_simulations: {len(all_phi_targets)}")


if __name__ == "__main__":
    main()

