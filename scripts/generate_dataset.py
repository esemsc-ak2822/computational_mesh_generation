import os
import sys
import time
import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from meshgen.naca4 import gen_naca4
from meshgen.domain import gen_outer_boundary, sample_farfield_points, build_point_cloud
from meshgen.triangulate import generate_mesh
from fem.laplace.solve import MeshGeometry, solve_laplace
from fem.common.basis_p1 import shape_function_gradients

def calculate_sdf(grid_points, airfoil_poly):
    """
    Calculate Signed Distance Field for a set of points relative to an airfoil polygon.
    Negative inside, positive outside.
    """
    sdf = np.zeros(len(grid_points))
    for i, (px, py) in enumerate(grid_points):
        p = Point(px, py)
        dist = p.distance(airfoil_poly)
        if airfoil_poly.contains(p):
            sdf[i] = -dist
        else:
            sdf[i] = dist
    return sdf

def run_simulation(naca_code, alpha_deg, v_inf=1.0):
    """
    Run a potential flow simulation for a given NACA code and angle of attack.
    Returns the geometry, results, and a dictionary of timing profiles.
    """
    profiles = {}
    
    alpha_rad = np.radians(alpha_deg)
    
    # 1. Generate Boundaries
    chord = 1.0
    airfoil_boundary = gen_naca4(naca_code, chord=chord, n_chord=120)
    outer_boundary = gen_outer_boundary(radius=5.0, n_points=120)
    
    # 2. Generate Mesh
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed=42)
    farfield_pts = sample_farfield_points(
        outer_boundary=outer_boundary,
        airfoil_boundary=airfoil_boundary,
        n_points=1500,
        rng=rng
    )
    
    points = build_point_cloud(
        airfoil_boundary=airfoil_boundary,
        outer_boundary=outer_boundary,
        farfield_points=farfield_pts
    )
    
    pts, triangles = generate_mesh(points, outer_boundary, airfoil_boundary)
    profiles['mesh_gen_sec'] = time.perf_counter() - t_start
    
    # 3. Solve Laplace
    t_start = time.perf_counter()
    def farfield_phi(x, y):
        return v_inf * (x * np.cos(alpha_rad) + y * np.sin(alpha_rad))
    
    geometry = MeshGeometry(
        points=pts,
        triangles=triangles,
        airfoil_boundary=airfoil_boundary,
        outer_boundary=outer_boundary
    )
    
    phi = solve_laplace(farfield_phi, geometry)
    profiles['fem_solve_sec'] = time.perf_counter() - t_start
    
    # 4. Compute Velocity and Pressure (Post-processing)
    t_start = time.perf_counter()
    n_nodes = pts.shape[0]
    n_tri = triangles.shape[0]
    
    velocity_tri = np.zeros((n_tri, 2))
    for i, tri in enumerate(triangles):
        coords = pts[tri]
        grads = shape_function_gradients(coords)
        grad_phi = phi[tri] @ grads
        velocity_tri[i] = grad_phi
        
    velocity_nodes = np.zeros((n_nodes, 2))
    counts = np.zeros(n_nodes)
    for i, tri in enumerate(triangles):
        velocity_nodes[tri] += velocity_tri[i]
        counts[tri] += 1
    
    velocity_nodes /= counts[:, None]
    
    v_mag_sq = np.sum(velocity_nodes**2, axis=1)
    pressure_nodes = 0.5 * (v_inf**2 - v_mag_sq)
    profiles['post_process_sec'] = time.perf_counter() - t_start
    
    return geometry, phi, velocity_nodes, pressure_nodes, profiles

def main():
    naca_airfoils = ["0012", "1410", "2412", "4412", "6412", "0015", "2415", "4415", "2410", "0010"]
    alphas = np.linspace(-5, 15, 10) # 10 alpha values from -5 to 15
    
    # Define sensors for Part 1: Potential on 100 points on the outer boundary
    sensor_theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    radius_sensors = 5.0
    bc_sensor_coords = np.column_stack([radius_sensors * np.cos(sensor_theta), radius_sensors * np.sin(sensor_theta)])
    
    # Define sensors for Part 2: SDF on a 32x32 grid
    gx = np.linspace(-1.0, 2.0, 32)
    gy = np.linspace(-1.0, 1.0, 32)
    GX, GY = np.meshgrid(gx, gy)
    sdf_sensor_coords = np.column_stack([GX.ravel(), GY.ravel()])
    
    # Data storage
    all_sdf_sensors = []
    all_bc_sensors = []
    all_phi_targets = []
    all_velocity_targets = []
    all_pressure_targets = []
    all_mesh_coords = []
    all_mesh_triangles = []
    
    # Also save metadata
    simulation_metadata = []

    print(f"Starting data generation for 10 airfoils and 10 inlet conditions...")

    for code in naca_airfoils:
        airfoil_poly = Polygon(gen_naca4(code, chord=1.0, n_chord=120))
        # Calculate SDF for this airfoil once (it doesn't change with alpha)
        sdf_vals = calculate_sdf(sdf_sensor_coords, airfoil_poly)
        
        for alpha in alphas:
            print(f"Running NACA {code} at Alpha = {alpha:.2f}")
            try:
                geometry, phi, velocity, pressure, profiles = run_simulation(code, alpha)
                
                # BC sensors (input values at sensor locations)
                alpha_rad = np.radians(alpha)
                bc_vals = 1.0 * (bc_sensor_coords[:, 0] * np.cos(alpha_rad) + bc_sensor_coords[:, 1] * np.sin(alpha_rad))
                
                all_sdf_sensors.append(sdf_vals)
                all_bc_sensors.append(bc_vals)
                all_phi_targets.append(phi)
                all_velocity_targets.append(velocity)
                all_pressure_targets.append(pressure)
                all_mesh_coords.append(geometry.points)
                all_mesh_triangles.append(geometry.triangles)
                
                # Combine metadata and profiles
                meta = {
                    "naca": code,
                    "alpha": alpha,
                    "n_nodes": len(phi)
                }
                meta.update(profiles)
                simulation_metadata.append(meta)
            except Exception as e:
                print(f"Failed simulation for {code} at {alpha}: {e}")

    # Convert lists to objects appropriate for npz
    # Since meshes have different numbers of points, we store them as lists of arrays
    # Deep learning friendly might mean fixed size, but for now we provide the raw mesh data
    # and the sensor data.
    
    save_path = Path("dataset_deeponet.npz")
    np.savez_compressed(
        save_path,
        sdf_sensors=np.array(all_sdf_sensors), # (N_sim, 1024)
        bc_sensors=np.array(all_bc_sensors),   # (N_sim, 100)
        phi_targets=np.array(all_phi_targets, dtype=object), # List of (N_nodes,)
        velocity_targets=np.array(all_velocity_targets, dtype=object), # List of (N_nodes, 2)
        pressure_targets=np.array(all_pressure_targets, dtype=object), # List of (N_nodes,)
        mesh_coords=np.array(all_mesh_coords, dtype=object), # List of (N_nodes, 2)
        mesh_triangles=np.array(all_mesh_triangles, dtype=object), # List of (N_tri, 3)
        metadata=simulation_metadata
    )
    
    print(f"Dataset generated and saved to {save_path}")

if __name__ == "__main__":
    main()
