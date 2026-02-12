import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from meshgen.domain import airfoil_offset_rings


def calculate_sdf(query_points: np.ndarray, airfoil_poly: Polygon) -> np.ndarray:
    sdf = np.zeros(len(query_points), dtype=float)
    for i, (px, py) in enumerate(query_points):
        p = Point(float(px), float(py))
        dist = p.distance(airfoil_poly)
        sdf[i] = -dist if airfoil_poly.contains(p) else dist
    return sdf


def build_sensor_layout(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    sensor_theta = np.linspace(0.0, 2.0 * np.pi, int(cfg["bc_sensor_count"]), endpoint=False)
    radius = float(cfg["bc_sensor_radius"])
    bc_sensor_coords = np.column_stack([radius * np.cos(sensor_theta), radius * np.sin(sensor_theta)])

    gx = np.linspace(float(cfg["sdf_xmin"]), float(cfg["sdf_xmax"]), int(cfg["sdf_nx"]))
    gy = np.linspace(float(cfg["sdf_ymin"]), float(cfg["sdf_ymax"]), int(cfg["sdf_ny"]))
    GX, GY = np.meshgrid(gx, gy)
    sdf_global_coords = np.column_stack([GX.ravel(), GY.ravel()])
    return bc_sensor_coords, sdf_global_coords


def build_near_airfoil_sensor_points(
    airfoil_boundary: np.ndarray,
    near_dists: list[float],
    per_ring: int,
) -> np.ndarray:
    rings_all = airfoil_offset_rings(airfoil_boundary, distances=near_dists)
    n_boundary_ring = rings_all.shape[0] // len(near_dists)
    ring_coords = []
    for ring_idx in range(len(near_dists)):
        ring = rings_all[ring_idx * n_boundary_ring : (ring_idx + 1) * n_boundary_ring]
        idx = np.linspace(0, n_boundary_ring - 1, per_ring, dtype=int)
        ring_coords.append(ring[idx])
    return np.vstack(ring_coords)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sensor features (SDF + BC) from FEM-only dataset.")
    parser.add_argument("--fem-input", type=Path, default=Path("dataset_fem_v1.npz"))
    parser.add_argument("--output", type=Path, default=Path("dataset_sensors_v1.npz"))
    parser.add_argument("--sdf-nx", type=int, default=48)
    parser.add_argument("--sdf-ny", type=int, default=48)
    parser.add_argument("--sdf-xmin", type=float, default=-1.0)
    parser.add_argument("--sdf-xmax", type=float, default=2.0)
    parser.add_argument("--sdf-ymin", type=float, default=-1.0)
    parser.add_argument("--sdf-ymax", type=float, default=1.0)
    parser.add_argument("--bc-sensors", type=int, default=180)
    parser.add_argument("--bc-radius", type=float, default=None)
    parser.add_argument("--near-offsets", type=str, default="0.005,0.01,0.02,0.03")
    parser.add_argument("--near-sensors-per-ring", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    fem = np.load(args.fem_input, allow_pickle=True)

    metadata = fem["metadata"]
    airfoil_boundaries = fem["airfoil_boundaries"]

    if "fem_config_json" in fem.files:
        fem_cfg = json.loads(str(fem["fem_config_json"].item()))
    else:
        fem_cfg = {}

    bc_radius = float(args.bc_radius) if args.bc_radius is not None else float(fem_cfg.get("outer_radius", 5.0))
    near_offsets = [float(x.strip()) for x in args.near_offsets.split(",") if x.strip()]

    cfg = {
        "dataset_version": "sensor_only_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_fem_path": str(args.fem_input),
        "source_fem_hash": str(fem["fem_config_hash"].item()) if "fem_config_hash" in fem.files else "unknown",
        "bc_sensor_count": int(args.bc_sensors),
        "bc_sensor_radius": bc_radius,
        "sdf_nx": int(args.sdf_nx),
        "sdf_ny": int(args.sdf_ny),
        "sdf_xmin": float(args.sdf_xmin),
        "sdf_xmax": float(args.sdf_xmax),
        "sdf_ymin": float(args.sdf_ymin),
        "sdf_ymax": float(args.sdf_ymax),
        "sdf_near_offsets": near_offsets,
        "sdf_near_per_ring": int(args.near_sensors_per_ring),
    }

    cfg_json = json.dumps(cfg, sort_keys=True)
    cfg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()[:12]
    cfg["config_hash"] = cfg_hash

    bc_sensor_coords, sdf_global_coords = build_sensor_layout(cfg)

    all_sdf_sensors = []
    all_sdf_global_sensors = []
    all_sdf_near_sensors = []
    all_bc_sensors = []
    sim_ids = []

    print(f"Generating sensor dataset from {args.fem_input}")
    print(f"Sensor config hash: {cfg_hash}")

    for i, meta in enumerate(metadata):
        alpha = float(meta["alpha"])
        v_inf = float(meta.get("v_inf", 1.0))
        airfoil_boundary = np.asarray(airfoil_boundaries[i], dtype=float)
        airfoil_poly = Polygon(airfoil_boundary)

        sdf_global = calculate_sdf(sdf_global_coords, airfoil_poly)
        sdf_near_coords = build_near_airfoil_sensor_points(
            airfoil_boundary=airfoil_boundary,
            near_dists=near_offsets,
            per_ring=int(cfg["sdf_near_per_ring"]),
        )
        sdf_near = calculate_sdf(sdf_near_coords, airfoil_poly)
        sdf_combined = np.concatenate([sdf_global, sdf_near], axis=0)

        alpha_rad = np.radians(alpha)
        bc_vals = v_inf * (
            bc_sensor_coords[:, 0] * np.cos(alpha_rad) + bc_sensor_coords[:, 1] * np.sin(alpha_rad)
        )

        all_sdf_sensors.append(sdf_combined)
        all_sdf_global_sensors.append(sdf_global)
        all_sdf_near_sensors.append(sdf_near)
        all_bc_sensors.append(bc_vals)
        sim_ids.append(int(meta["sim_id"]) if "sim_id" in meta else i)

    np.savez_compressed(
        args.output,
        sdf_sensors=np.array(all_sdf_sensors, dtype=float),
        bc_sensors=np.array(all_bc_sensors, dtype=float),
        sdf_global_sensors=np.array(all_sdf_global_sensors, dtype=float),
        sdf_near_sensors=np.array(all_sdf_near_sensors, dtype=float),
        bc_sensor_coords=bc_sensor_coords,
        sdf_global_sensor_coords=sdf_global_coords,
        sdf_near_sensor_offsets=np.array(near_offsets, dtype=float),
        sim_ids=np.array(sim_ids, dtype=int),
        sensor_config_json=np.array(cfg_json),
        sensor_config_hash=np.array(cfg_hash),
    )

    config_sidecar = args.output.with_suffix(".meta.json")
    config_sidecar.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Sensor dataset generated and saved to {args.output}")
    print(f"Sensor config metadata saved to {config_sidecar}")
    print(f"sdf_sensors shape: {np.array(all_sdf_sensors, dtype=float).shape}")
    print(f"bc_sensors shape: {np.array(all_bc_sensors, dtype=float).shape}")


if __name__ == "__main__":
    main()

