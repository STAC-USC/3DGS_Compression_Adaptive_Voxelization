import os
import numpy as np
import open3d as o3d
import argparse
import shutil
from scene.gaussian_model import GaussianModel
from plyfile import PlyData, PlyElement
import torch


def load_ply_3DGS(file_path):
    """
    Load a PLY file containing the following attributes without normalization or reduction:
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    property float f_dc_0
    property float f_dc_1
    property float f_dc_2
    property float f_rest_0
    ...
    property float f_rest_44
    property float opacity
    property float scale_0
    property float scale_1
    property float scale_2
    property float rot_0
    property float rot_1
    property float rot_2
    property float rot_3
    """
    plydata = PlyData.read(file_path)
    vertex_data = plydata['vertex']

    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    nx = vertex_data['nx']
    ny = vertex_data['ny']
    nz = vertex_data['nz']
    f_dc_0 = vertex_data['f_dc_0']
    f_dc_1 = vertex_data['f_dc_1']
    f_dc_2 = vertex_data['f_dc_2']

    # f_rest_0 ~ f_rest_44 
    f_rest = np.column_stack([vertex_data[f'f_rest_{i}'] for i in range(45)])

    opacity = vertex_data['opacity']
    scale_0 = vertex_data['scale_0']
    scale_1 = vertex_data['scale_1']
    scale_2 = vertex_data['scale_2']
    rot_0 = vertex_data['rot_0']
    rot_1 = vertex_data['rot_1']
    rot_2 = vertex_data['rot_2']
    rot_3 = vertex_data['rot_3']

    points = np.column_stack((x, y, z))
    normals = np.column_stack((nx, ny, nz))
    f_dc = np.column_stack((f_dc_0, f_dc_1, f_dc_2))
    scales = np.column_stack((scale_0, scale_1, scale_2))
    rots = np.column_stack((rot_0, rot_1, rot_2, rot_3))

    return points, normals, f_dc, f_rest, opacity, scales, rots


def save_ply_xyz(filename, points):
    """
    Save a PLY file containing x, y, z (float) in binary format.
    """
    # Build a structured array (x, y, z)
    # Define data type
    vertex_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
    ])

    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices['x'] = points[:, 0]
    vertices['y'] = points[:, 1]
    vertices['z'] = points[:, 2]

    # Create PlyElement using the plyfile library
    el = PlyElement.describe(vertices, 'vertex')
    # Write using binary_little_endian format
    PlyData([el], byte_order='<', text=False).write(filename)


def save_ply_3DGS(filename, points, normals, f_dc, f_rest, opacity, scales, rots):
    """
    Save a PLY file containing all attributes corresponding to those loaded, in binary format:
    x, y, z
    nx, ny, nz
    f_dc_0, f_dc_1, f_dc_2
    f_rest_0 ... f_rest_44
    opacity
    scale_0, scale_1, scale_2
    rot_0, rot_1, rot_2, rot_3
    """
    num_points = points.shape[0]
    # Define data type
    # f_rest has 45 attributes
    f_rest_types = [('f_rest_%d' % i, np.float32) for i in range(45)]
    vertex_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('nx', np.float32),
        ('ny', np.float32),
        ('nz', np.float32),
        ('f_dc_0', np.float32),
        ('f_dc_1', np.float32),
        ('f_dc_2', np.float32),
    ] + f_rest_types + [
        ('opacity', np.float32),
        ('scale_0', np.float32),
        ('scale_1', np.float32),
        ('scale_2', np.float32),
        ('rot_0', np.float32),
        ('rot_1', np.float32),
        ('rot_2', np.float32),
        ('rot_3', np.float32)
    ])

    vertices = np.empty(num_points, dtype=vertex_dtype)
    vertices['x'] = points[:, 0]
    vertices['y'] = points[:, 1]
    vertices['z'] = points[:, 2]

    vertices['nx'] = normals[:, 0]
    vertices['ny'] = normals[:, 1]
    vertices['nz'] = normals[:, 2]

    vertices['f_dc_0'] = f_dc[:, 0]
    vertices['f_dc_1'] = f_dc[:, 1]
    vertices['f_dc_2'] = f_dc[:, 2]

    for i in range(45):
        vertices['f_rest_%d' % i] = f_rest[:, i]

    vertices['opacity'] = opacity
    vertices['scale_0'] = scales[:, 0]
    vertices['scale_1'] = scales[:, 1]
    vertices['scale_2'] = scales[:, 2]

    vertices['rot_0'] = rots[:, 0]
    vertices['rot_1'] = rots[:, 1]
    vertices['rot_2'] = rots[:, 2]
    vertices['rot_3'] = rots[:, 3]

    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], byte_order='<', text=False).write(filename)


def save_ply_PC(filename, final_points, final_f_dc):
    """
    Normalize final_f_dc from float32 to uchar 0~255 and save as a PLY file.
    File name format: {output_filename_base}_PC.ply
    Color attributes are named red, green, blue.
    """
    # Normalize final_f_dc to 0~255, assuming shape is (N, 3)
    min_val = np.min(final_f_dc)
    max_val = np.max(final_f_dc)
    if max_val - min_val < 1e-6:
        normalized_f_dc = np.zeros_like(final_f_dc, dtype=np.uint8)
    else:
        normalized_f_dc = ((final_f_dc - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    vertex_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('red', np.uint8),
        ('green', np.uint8),
        ('blue', np.uint8),
    ])
    vertices = np.empty(len(final_points), dtype=vertex_dtype)
    vertices['x'] = final_points[:, 0]
    vertices['y'] = final_points[:, 1]
    vertices['z'] = final_points[:, 2]
    vertices['red'] = normalized_f_dc[:, 0]
    vertices['green'] = normalized_f_dc[:, 1]
    vertices['blue'] = normalized_f_dc[:, 2]

    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], byte_order='<', text=False).write(filename)



def get_morton_code(V, J):
    V = V.astype(np.uint64)
    N = V.shape[0]
    M = np.zeros(N, dtype=np.uint64)

    for i in range(J):
        bits = (V >> i) & 1
        M |= (bits[:, 2] << (3 * i + 0)) | (bits[:, 1] << (3 * i + 1)) | (bits[:, 0] << (3 * i + 2))
    return M


def adaptive_voxelization(points, normals, f_dc, f_rest, scales, rots, depth_start, voxel_thr, gaussian_ply):
    """
    Perform adaptive voxelization, and recalculate attributes (recoloring) in the final deduplication stage.
    """
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(gaussian_ply)  # Load using the given gaussian_ply path

    depth = depth_start
    min_coords = np.min(points, axis=0)
    bounding_box_size = np.max(points - min_coords)
    if bounding_box_size == 0:
        bounding_box_size = 1  # Avoid division by zero

    total_points = points.shape[0]
    print(f"[DEBUG] Starting adaptive voxelization with depth_start={depth_start}, voxel_thr={voxel_thr}")

    # Get activated scaling and activated opacity to compute volume and opacity
    activated_scaling = gaussians.get_scaling.detach().cpu().numpy()  # Nx3
    volumes = np.prod(activated_scaling, axis=1)  # N
    # Used later to identify large Gaussian points
    volume_threshold = np.percentile(volumes, 80) 
    large_gaussian_mask = volumes >= volume_threshold

    moved_mask = np.zeros(total_points, dtype=bool)

    while True:
        voxel_size = bounding_box_size / (2 ** depth)
        voxel_indices = np.floor((points - min_coords) / voxel_size).astype(np.int64)

        unique_voxels, inverse_indices, counts = np.unique(
            voxel_indices, axis=0, return_inverse=True, return_counts=True
        )
        print(f"[DEBUG] Depth={depth}: unique voxels={len(unique_voxels)}, "
              f"voxels with <{voxel_thr} points={np.sum(counts < voxel_thr)}")

        # Find small voxels
        small_voxel_mask = counts < voxel_thr
        small_voxel_indices = unique_voxels[small_voxel_mask]
        small_voxel_centers = (small_voxel_indices + 0.5) * voxel_size + min_coords

        small_voxel_map_indices = np.where(small_voxel_mask)[0]
        full_to_small_map = {idx: i for i, idx in enumerate(small_voxel_map_indices)}

        small_voxel_map = np.isin(inverse_indices, small_voxel_map_indices)
        points_to_check = np.where(small_voxel_map & ~moved_mask)[0]

        if len(points_to_check) > 0:
            next_voxel_size = bounding_box_size / (2 ** (depth + 1))
            small_voxel_center_indices = [full_to_small_map[idx] for idx in inverse_indices[points_to_check]]
            target_centers = small_voxel_centers[small_voxel_center_indices]

            # Distance criteria
            distances = np.linalg.norm(points[points_to_check] - target_centers, axis=1)
            eligible_to_move = distances < next_voxel_size

            # eligible_to_move = np.ones_like(points_to_check, dtype=bool)

            eligible_indices = points_to_check[eligible_to_move]
            large_gaussian_indices = eligible_indices[large_gaussian_mask[eligible_indices]]
            non_large_indices = np.setdiff1d(eligible_indices, large_gaussian_indices)

            if len(non_large_indices) > 0:
                non_large_center_indices = [full_to_small_map[idx] for idx in inverse_indices[non_large_indices]]
                points[non_large_indices] = small_voxel_centers[non_large_center_indices]
                moved_mask[non_large_indices] = True

            moved_mask[large_gaussian_indices] = False

        remaining_large_voxels = np.sum(~small_voxel_mask)
        if remaining_large_voxels == 0:
            depth += 1
            print(f"[DEBUG] Adaptive voxelization completed at depth={depth}")
            break

        depth += 1

    # Final deduplication and attribute handling
    voxel_size = bounding_box_size / (2 ** depth)
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(np.int64)
    unique_voxel_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    num_voxels = len(unique_voxel_indices)

    # Prepare for aggregation of all attributes
    # Attributes to be averaged: normals, f_dc, f_rest, rots
    # Attributes to be handled specially: scales (use raw scale of max volume point), opacity (sum activated, then inverse), rot (average)
    # Clarified: rot is averaged, scales use max volume point, f_dc and f_rest averaged, normals averaged, opacity special handling

    # Accumulators
    sum_normals = np.zeros((num_voxels, 3), dtype=np.float64)
    sum_f_dc = np.zeros((num_voxels, 3), dtype=np.float64)
    sum_f_rest = np.zeros((num_voxels, 45), dtype=np.float64)
    sum_rot = np.zeros((num_voxels, 4), dtype=np.float64)
    voxel_counts = np.zeros(num_voxels, dtype=np.float64)

    # Special handling for scale and opacity
    # For scale: find the point with the max volume
    # For opacity: sum activated_opacity, then inverse
    activated_opacity = gaussians.get_opacity.detach().cpu().numpy()  # Nx1

    # Indices list for each voxel
    voxel_point_indices = [[] for _ in range(num_voxels)]
    for i, v in enumerate(inverse_indices):
        voxel_point_indices[v].append(i)

    final_scales = np.zeros((num_voxels, 3), dtype=np.float32)
    # opacity needs activated_opacity aggregation before inversion
    sum_activated_opacity = np.zeros(num_voxels, dtype=np.float64)

    for v in range(num_voxels):
        pts_idx = voxel_point_indices[v]
        if len(pts_idx) == 0:
            continue
        voxel_counts[v] = len(pts_idx)

        # Average normals, f_dc, f_rest, rot
        sum_normals[v] = np.sum(normals[pts_idx], axis=0)
        sum_f_dc[v] = np.sum(f_dc[pts_idx], axis=0)
        sum_f_rest[v] = np.sum(f_rest[pts_idx], axis=0)
        sum_rot[v] = np.sum(rots[pts_idx], axis=0)
        sum_activated_opacity[v] = np.mean(activated_opacity[pts_idx])

        # Get the scale of the max volume point
        voxel_volumes = volumes[pts_idx]
        max_vol_idx = pts_idx[np.argmax(voxel_volumes)]
        # Use raw scale of this point as the final voxel scale
        final_scales[v] = scales[max_vol_idx]

    # Averaging
    final_normals = (sum_normals / voxel_counts[:, None]).astype(np.float32)
    final_f_dc = (sum_f_dc / voxel_counts[:, None]).astype(np.float32)
    final_f_rest = (sum_f_rest / voxel_counts[:, None]).astype(np.float32)
    final_rot = (sum_rot / voxel_counts[:, None]).astype(np.float32)

    # Opacity processing: inverse activated_opacity to get raw opacity
    # Note: gaussians.inverse_opacity_activation expects torch.tensor(denormalized_opacity)
    # sum_activated_opacity is the accumulated activated value, to be inverted
    # User requirement: sum activated opacity, then convert to raw data using inverse_opacity_activation
    sum_activated_opacity_tensor = torch.tensor(sum_activated_opacity, dtype=torch.float32)
    final_opacity = gaussians.inverse_opacity_activation(sum_activated_opacity_tensor).numpy().astype(np.float32)

    # Compute final_points
    final_points = (unique_voxel_indices.astype(np.float32) + 0.5) * voxel_size + min_coords

    # Morton sort
    max_voxel_coord = np.max(unique_voxel_indices)
    J = int(np.ceil(np.log2(max_voxel_coord + 1))) if max_voxel_coord > 0 else 1
    morton_codes = get_morton_code(unique_voxel_indices, J)
    sort_idx = np.argsort(morton_codes)

    final_points = final_points[sort_idx]
    final_normals = final_normals[sort_idx]
    final_f_dc = final_f_dc[sort_idx]
    final_f_rest = final_f_rest[sort_idx]
    final_opacity = final_opacity[sort_idx]
    final_scales = final_scales[sort_idx]
    final_rot = final_rot[sort_idx]

    # Also sort unique_voxel_indices for xyz output
    sorted_unique_voxel_indices = unique_voxel_indices[sort_idx].astype(np.float32)

    # Debug output before return:
    print("[DEBUG] Checking for NaNs in recoloring results:")
    # Check opacity
    num_nan_opacity = np.isnan(final_opacity).sum()
    if num_nan_opacity > 0:
        print(f"[WARNING] final_opacity has {num_nan_opacity} NaNs (total {final_opacity.size} elements).")
    else:
        print("[DEBUG] final_opacity has no NaNs.")

    # Check scales
    num_nan_scales = np.isnan(final_scales).sum()
    if num_nan_scales > 0:
        print(f"[WARNING] final_scales has {num_nan_scales} NaNs (total {final_scales.size} elements).")
    else:
        print("[DEBUG] final_scales has no NaNs.")

    # Check rot
    num_nan_rot = np.isnan(final_rot).sum()
    if num_nan_rot > 0:
        print(f"[WARNING] final_rot has {num_nan_rot} NaNs (total {final_rot.size} elements).")
    else:
        print("[DEBUG] final_rot has no NaNs.")

    print(f"[DEBUG] Final voxelization completed: points={len(final_points)}")

    return final_points, final_normals, final_f_dc, final_f_rest, final_opacity, final_scales, final_rot, depth, sorted_unique_voxel_indices




# Uniform Voxelization
def uniform_voxelization(points, normals, f_dc, f_rest, scales, rots, depth_start, voxel_thr, gaussian_ply):
    """
    Perform fixed-depth (uniform) voxelization and recalculate attributes (recoloring) during the final deduplication stage.
    The depth will no longer be increased dynamically based on voxel_thr, but instead uses the provided depth_start directly.
    """

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(gaussian_ply)  # Load using the given gaussian_ply path

    depth = depth_start
    min_coords = np.min(points, axis=0)
    bounding_box_size = np.max(points - min_coords)
    if bounding_box_size == 0:
        bounding_box_size = 1  # Avoid division by zero

    print(f"[DEBUG] Starting uniform voxelization with fixed depth={depth}, voxel_thr={voxel_thr}")

    # ============== Removed the previous "adaptive" part, only keeping one-time calculation ===============
    voxel_size = bounding_box_size / (2 ** depth)
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(np.int64)
    # Deduplication
    unique_voxel_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    num_voxels = len(unique_voxel_indices)

    # ============== Following is the logic from the original "final deduplication and attribute processing", all retained ================
    # Prepare for aggregating all attributes
    activated_scaling = gaussians.get_scaling.detach().cpu().numpy()  # Nx3
    volumes = np.prod(activated_scaling, axis=1)  # N
    activated_opacity = gaussians.get_opacity.detach().cpu().numpy()  # Nx1

    # List of indices within each voxel
    voxel_point_indices = [[] for _ in range(num_voxels)]
    for i, v in enumerate(inverse_indices):
        voxel_point_indices[v].append(i)

    # Accumulators
    sum_normals = np.zeros((num_voxels, 3), dtype=np.float64)
    sum_f_dc = np.zeros((num_voxels, 3), dtype=np.float64)
    sum_f_rest = np.zeros((num_voxels, 45), dtype=np.float64)
    sum_rot = np.zeros((num_voxels, 4), dtype=np.float64)
    voxel_counts = np.zeros(num_voxels, dtype=np.float64)

    final_scales = np.zeros((num_voxels, 3), dtype=np.float32)
    sum_activated_opacity = np.zeros(num_voxels, dtype=np.float64)

    for v in range(num_voxels):
        pts_idx = voxel_point_indices[v]
        if len(pts_idx) == 0:
            continue
        voxel_counts[v] = len(pts_idx)

        # Average normals, f_dc, f_rest, rot
        sum_normals[v] = np.sum(normals[pts_idx], axis=0)
        sum_f_dc[v] = np.sum(f_dc[pts_idx], axis=0)
        sum_f_rest[v] = np.sum(f_rest[pts_idx], axis=0)
        sum_rot[v] = np.sum(rots[pts_idx], axis=0)
        # Here we use the logic of averaging activated opacity inside each voxel
        sum_activated_opacity[v] = np.mean(activated_opacity[pts_idx])

        # Find the scale of the point with the maximum volume
        voxel_volumes = volumes[pts_idx]
        max_vol_idx = pts_idx[np.argmax(voxel_volumes)]
        final_scales[v] = scales[max_vol_idx]  # Directly use the raw scale of that point

    # Averaging
    final_normals = (sum_normals / voxel_counts[:, None]).astype(np.float32)
    final_f_dc = (sum_f_dc / voxel_counts[:, None]).astype(np.float32)
    final_f_rest = (sum_f_rest / voxel_counts[:, None]).astype(np.float32)
    final_rot = (sum_rot / voxel_counts[:, None]).astype(np.float32)

    # Opacity processing: convert the averaged activated_opacity back to raw opacity
    sum_activated_opacity_tensor = torch.tensor(sum_activated_opacity, dtype=torch.float32)
    final_opacity = gaussians.inverse_opacity_activation(sum_activated_opacity_tensor).numpy().astype(np.float32)

    # Calculate final_points (voxel centers)
    final_points = (unique_voxel_indices.astype(np.float32) + 0.5) * voxel_size + min_coords

    # Morton sorting
    max_voxel_coord = np.max(unique_voxel_indices)
    J = int(np.ceil(np.log2(max_voxel_coord + 1))) if max_voxel_coord > 0 else 1
    morton_codes = get_morton_code(unique_voxel_indices, J)
    sort_idx = np.argsort(morton_codes)

    final_points = final_points[sort_idx]
    final_normals = final_normals[sort_idx]
    final_f_dc = final_f_dc[sort_idx]
    final_f_rest = final_f_rest[sort_idx]
    final_opacity = final_opacity[sort_idx]
    final_scales = final_scales[sort_idx]
    final_rot = final_rot[sort_idx]

    # Also sort unique_voxel_indices for saving xyz files
    sorted_unique_voxel_indices = unique_voxel_indices[sort_idx].astype(np.float32)

    # NaN check
    print("[DEBUG] Checking for NaNs in recoloring results:")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive voxelization and processing of point clouds')
    parser.add_argument('--depth_start', type=int, required=True, help='Initial depth for voxelization')
    parser.add_argument('--voxel_thr', type=int, required=True, help='Threshold for small voxel point count')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--use_adaptive', type=str, required=True, help='Set to "true" for adaptive voxelization, "false" for uniform voxelization')
    args = parser.parse_args()

    # Automatically get the current script directory, go up two levels as the root path
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    print(f"[DEBUG] Root path determined as: {root_path}")

    # Construct the input file path (based on root path)
    input_ply_path = os.path.join(
        root_path,
        'original_model',
        args.dataset_name,
        'point_cloud',
        'iteration_30000',
        'point_cloud.ply'
    )

    # gaussian_ply is no longer passed via CLI, but constructed as original_model/{dataset_name}/point_cloud/iteration_30000/point_cloud.ply under the root directory
    base_ply_path = os.path.join(root_path, 'original_model', '{dataset_name}', 'point_cloud', 'iteration_30000', 'point_cloud.ply')
    gaussian_ply = base_ply_path.format(dataset_name=args.dataset_name)

    # output_path is fixed as voxelized_adapt under the root directory
    output_path = os.path.join(root_path, 'voxelized_adapt')

    if not os.path.exists(input_ply_path):
        raise FileNotFoundError(f"Input PLY file not found at path: {input_ply_path}")

    # Load point cloud and all attributes
    points, normals, f_dc, f_rest, opacity, scales, rots = load_ply_3DGS(input_ply_path)
    print(f"[DEBUG] Starting voxelization for dataset '{args.dataset_name}' with depth_start={args.depth_start}")

    # Choose adaptive or fixed-depth voxelization based on arguments
    if args.use_adaptive.lower() == "true":
        final_points, final_normals, final_f_dc, final_f_rest, final_opacity, final_scales, final_rot, final_depth, sorted_unique_voxel_indices = adaptive_voxelization(
            points, normals, f_dc, f_rest, scales, rots, args.depth_start, args.voxel_thr, gaussian_ply
        )
        suffix = "_adapt"
    else:
        final_points, final_normals, final_f_dc, final_f_rest, final_opacity, final_scales, final_rot, final_depth, sorted_unique_voxel_indices = uniform_voxelization(
            points, normals, f_dc, f_rest, scales, rots, args.depth_start, args.voxel_thr, gaussian_ply
        )
        suffix = "_uniform"

    output_filename_base = f"{args.dataset_name}_depth_{args.depth_start}_thr_{args.voxel_thr}"
    xyz_ply = os.path.join(output_path, f"{output_filename_base}_voxel_xyz{suffix}.ply")
    full_ply = os.path.join(output_path, f"{output_filename_base}_3DGS{suffix}.ply")
    pc_ply = os.path.join(output_path, f"{output_filename_base}_PC{suffix}.ply")

    # Save results
    save_ply_xyz(xyz_ply, sorted_unique_voxel_indices)
    save_ply_3DGS(full_ply, final_points, final_normals, final_f_dc, final_f_rest, final_opacity, final_scales, final_rot)
    save_ply_PC(pc_ply, final_points, final_f_dc)

    print(f"[DEBUG] Saved voxelized coordinates to {xyz_ply}")
    print(f"[DEBUG] Saved final 3DGS ply to {full_ply}")
    print(f"[DEBUG] Saved point cloud PC ply to {pc_ply}")

