import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge FEM-only dataset and sensor-only dataset into DeepONet-ready NPZ.")
    parser.add_argument("--fem-input", type=Path, default=Path("dataset_fem_v1.npz"))
    parser.add_argument("--sensor-input", type=Path, default=Path("dataset_sensors_v1.npz"))
    parser.add_argument("--output", type=Path, default=Path("dataset_deeponet_modular_v1.npz"))
    return parser.parse_args()


def main():
    args = parse_args()

    fem = np.load(args.fem_input, allow_pickle=True)
    sensors = np.load(args.sensor_input, allow_pickle=True)

    n_fem = len(fem["metadata"])
    n_sens = sensors["sdf_sensors"].shape[0]
    if n_fem != n_sens:
        raise ValueError(f"Simulation count mismatch: fem={n_fem}, sensors={n_sens}")

    if "sim_ids" in sensors.files:
        sim_ids = sensors["sim_ids"].astype(int)
        expected = np.arange(n_fem, dtype=int)
        if sim_ids.shape != expected.shape or not np.array_equal(sim_ids, expected):
            raise ValueError("Sensor sim_ids do not match FEM simulation ordering.")

    np.savez_compressed(
        args.output,
        sdf_sensors=sensors["sdf_sensors"],
        bc_sensors=sensors["bc_sensors"],
        sdf_global_sensors=sensors["sdf_global_sensors"],
        sdf_near_sensors=sensors["sdf_near_sensors"],
        bc_sensor_coords=sensors["bc_sensor_coords"],
        sdf_global_sensor_coords=sensors["sdf_global_sensor_coords"],
        sdf_near_sensor_offsets=sensors["sdf_near_sensor_offsets"],
        phi_targets=fem["phi_targets"],
        velocity_targets=fem["velocity_targets"],
        pressure_targets=fem["pressure_targets"],
        mesh_coords=fem["mesh_coords"],
        mesh_triangles=fem["mesh_triangles"],
        airfoil_boundaries=fem["airfoil_boundaries"],
        outer_boundaries=fem["outer_boundaries"],
        metadata=fem["metadata"],
        fem_config_json=fem["fem_config_json"] if "fem_config_json" in fem.files else np.array(""),
        fem_config_hash=fem["fem_config_hash"] if "fem_config_hash" in fem.files else np.array(""),
        sensor_config_json=sensors["sensor_config_json"] if "sensor_config_json" in sensors.files else np.array(""),
        sensor_config_hash=sensors["sensor_config_hash"] if "sensor_config_hash" in sensors.files else np.array(""),
    )

    print(f"Merged dataset saved to {args.output}")
    print(f"sdf_sensors shape: {sensors['sdf_sensors'].shape}")
    print(f"bc_sensors shape: {sensors['bc_sensors'].shape}")
    print(f"num simulations: {n_fem}")


if __name__ == "__main__":
    main()

