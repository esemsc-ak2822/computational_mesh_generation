import numpy as np

data = np.load("dataset_deeponet.npz", allow_pickle=True)
print("Keys in dataset:", data.files)
print("sdf_sensors shape:", data["sdf_sensors"].shape)
print("bc_sensors shape:", data["bc_sensors"].shape)
print("phi_targets shape:", data["phi_targets"].shape)
print("mesh_coords shape:", data["mesh_coords"].shape)

# Check one entry
print("First simulation metadata:", data["metadata"][0])
print("First phi_target length:", len(data["phi_targets"][0]))
