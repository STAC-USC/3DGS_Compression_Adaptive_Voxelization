import os
import numpy as np
import open3d as o3d
import shutil
import argparse
from plyfile import PlyData
import subprocess
import sys


def voxelize_and_recoloring(depth, input_file, PC, PC_voxelized, output_path, dataset_name, voxel_thr=50):
    """
    Performs adaptive voxelization and recoloring, saving outputs according to the specified file management structure.
    Args:
        depth: Initial depth for adaptive voxelization.
        input_file: Path to the input PLY file.
        PC: Path to save the float-coordinate PLY file.
        PC_voxelized: Path to save the voxelized (integer-coordinate) PLY file.
        output_path: Directory to save output files.
        dataset_name: Name of the dataset being processed.
        voxel_thr: Threshold for small voxel point count in adaptive voxelization.
    """
    # Read point cloud data with colors
    points, colors = load_ply_with_colors(input_file)

    # Perform adaptive voxelization
    print(f"[DEBUG] Starting adaptive voxelization for depth={depth}, dataset={dataset_name}")
    final_points, final_colors, final_depth, unique_voxel_indices = adaptive_voxelization(
        points, colors, depth, voxel_thr=5
    )

    # Compute min_coords and voxel size for saving metadata
    min_coords = np.min(points, axis=0)
    bounding_box_size = np.max(points - min_coords)
    final_voxel_size = bounding_box_size / (2 ** final_depth)

    # Save voxelized point cloud (integer coordinates)
    save_ply_with_colors(PC_voxelized, unique_voxel_indices.astype(np.float32), final_colors)

    # Save final points (floating-point coordinates)
    save_ply_with_colors(PC, final_points.astype(np.float32), final_colors)

    # Save metadata
    np.save(os.path.join(output_path, f"min_coords_{dataset_name}_depth_{final_depth}.npy"), min_coords)
    np.save(os.path.join(output_path, f"voxel_size_{dataset_name}_depth_{final_depth}.npy"), final_voxel_size)
    np.save(os.path.join(output_path, f"depth_{dataset_name}_depth_{final_depth}.npy"), np.array([final_depth]))

    print(f"[DEBUG] Adaptive voxelization completed for depth={final_depth}.")
    print(f"[INFO] Saved voxelized PLY to {PC_voxelized}")
    print(f"[INFO] Saved float-coordinate PLY to {PC}")


def adaptive_voxelization(points, colors, depth_start, voxel_thr):
    """
    Perform adaptive voxelization, merging small voxels at each depth level, and finalize with Morton ordering.
    Args:
        points: Nx3 NumPy array of point cloud coordinates.
        colors: Nx3 NumPy array of color values.
        depth_start: Initial depth level for voxelization.
        voxel_thr: Threshold for the maximum number of points in a "small" voxel.
    Returns:
        final_points: Deduplicated point cloud (Nx3 array).
        final_colors: Recolored point cloud (Nx3 array).
        final_depth: Final depth level reached during adaptive voxelization.
    """
    depth = depth_start
    min_coords = np.min(points, axis=0)
    bounding_box_size = np.max(points - min_coords)
    if bounding_box_size == 0:
        bounding_box_size = 1  # Avoid division by zero

    total_points = points.shape[0]
    print(f"[DEBUG] Starting adaptive voxelization with depth_start={depth_start}, voxel_thr={voxel_thr}")

    # Keep track of points that have already been moved
    moved_mask = np.zeros(total_points, dtype=bool)  # Track whether a point has been moved

    while True:
        # Compute voxelization at the current depth
        voxel_size = bounding_box_size / (2 ** depth)
        voxel_indices = np.floor((points - min_coords) / voxel_size).astype(np.int64)

        # Group points by voxel
        unique_voxels, inverse_indices, counts = np.unique(
            voxel_indices, axis=0, return_inverse=True, return_counts=True
        )
        print(f"[DEBUG] Depth={depth}: unique voxels={len(unique_voxels)}, "
              f"voxels with <{voxel_thr} points={np.sum(counts < voxel_thr)}")

        # Identify small voxels
        small_voxel_mask = counts < voxel_thr
        small_voxel_indices = unique_voxels[small_voxel_mask]
        small_voxel_centers = (small_voxel_indices + 0.5) * voxel_size + min_coords

        # Map full voxel indices to small voxel indices
        small_voxel_map_indices = np.where(small_voxel_mask)[0]
        full_to_small_map = {idx: i for i, idx in enumerate(small_voxel_map_indices)}

        # Find points in small voxels
        small_voxel_map = np.isin(inverse_indices, small_voxel_map_indices)
        points_to_check = np.where(small_voxel_map & ~moved_mask)[0]  # Points not moved yet

        if len(points_to_check) > 0:
            # Compute new voxel size for J+1
            next_voxel_size = bounding_box_size / (2 ** (depth + 1))

            # Map points to their corresponding small voxel centers
            small_voxel_center_indices = [full_to_small_map[idx] for idx in inverse_indices[points_to_check]]
            target_centers = small_voxel_centers[small_voxel_center_indices]

            # Calculate distances between original points and target centers
            distances = np.linalg.norm(points[points_to_check] - target_centers, axis=1)

            # Identify points eligible for movement (distance < next_voxel_size)
            eligible_to_move = distances < next_voxel_size

            # Update points for those eligible to move
            points[points_to_check[eligible_to_move]] = target_centers[eligible_to_move]

            # Mark moved points
            moved_mask[points_to_check[eligible_to_move]] = True

        # Check stopping condition
        remaining_large_voxels = np.sum(~small_voxel_mask)
        if remaining_large_voxels == 0:
            depth += 1
            print(f"[DEBUG] Adaptive voxelization completed at depth={depth}")
            break

        depth += 1

    # Final deduplication, recoloring, and Morton sorting
    voxel_size = bounding_box_size / (2 ** depth)
    voxel_indices = np.floor((points - min_coords) / voxel_size).astype(np.int64)

    unique_voxel_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    final_colors = compute_average_colors(colors, inverse_indices, len(unique_voxel_indices))
    final_points = (unique_voxel_indices.astype(np.float32) + 0.5) * voxel_size + min_coords

    # Morton sorting
    max_voxel_coord = np.max(unique_voxel_indices)
    J = int(np.ceil(np.log2(max_voxel_coord + 1)))
    morton_codes = get_morton_code(unique_voxel_indices, J)
    sort_idx = np.argsort(morton_codes)

    final_points = final_points[sort_idx]
    final_colors = final_colors[sort_idx]

    print(f"[DEBUG] Final voxelization completed: points={len(final_points)}")
    return final_points, final_colors, depth, unique_voxel_indices


def load_ply_with_colors(file_path):
    # Read PLY file using plyfile
    plydata = PlyData.read(file_path)

    # Extract x, y, z, f_dc_0, f_dc_1, f_dc_2
    vertex_data = plydata['vertex']
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    f_dc_0 = vertex_data['f_dc_0']
    f_dc_1 = vertex_data['f_dc_1']
    f_dc_2 = vertex_data['f_dc_2']

    # Combine x, y, z into points array
    points = np.vstack((x, y, z)).T

    # Combine f_dc_0, f_dc_1, f_dc_2 into colors array
    f_dc = np.vstack((f_dc_0, f_dc_1, f_dc_2)).T

    # Linearly normalize f_dc_0, f_dc_1, f_dc_2 to [0, 255]
    min_vals = np.min(f_dc, axis=0)
    max_vals = np.max(f_dc, axis=0)
    denom = max_vals - min_vals
    denom[denom == 0] = 1  # Avoid division by zero
    f_dc_normalized = (f_dc - min_vals) / denom
    colors = (f_dc_normalized * 255).astype(np.uint8)

    return points, colors

def save_ply_with_colors(filename, points, colors):
    # Save point cloud using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

    # Modify the header to change data types from 'double' to 'float'
    with open(filename, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if 'property double x' in line:
            new_lines.append('property float x\n')
        elif 'property double y' in line:
            new_lines.append('property float y\n')
        elif 'property double z' in line:
            new_lines.append('property float z\n')
        else:
            new_lines.append(line)

    with open(filename, 'w') as file:
        file.writelines(new_lines)

def compute_average_colors(colors, point_to_voxel_map, num_voxels):
    voxel_colors_sum = np.zeros((num_voxels, 3), dtype=np.float64)
    voxel_counts = np.zeros(num_voxels, dtype=np.float64)
    np.add.at(voxel_colors_sum, point_to_voxel_map, colors)
    np.add.at(voxel_counts, point_to_voxel_map, 1)
    voxel_colors = voxel_colors_sum / voxel_counts[:, None]
    voxel_colors = voxel_colors.astype(np.uint8)
    return voxel_colors

def get_morton_code(V, J):
    # V is an Nx3 array of integer coordinates
    V = V.astype(np.uint64)
    N = V.shape[0]
    M = np.zeros(N, dtype=np.uint64)

    for i in range(J):
        bits = (V >> i) & 1  # Extract i-th bit of each coordinate
        M |= (bits[:, 2] << (3 * i + 0)) | (bits[:, 1] << (3 * i + 1)) | (bits[:, 0] << (3 * i + 2))
    return M

def float_to_int_xyz(points_float):
    try:
        if not hasattr(float_to_int_xyz, 'min_coords'):
            # 检查文件是否存在
            if not os.path.exists('min_coords.npy') or not os.path.exists('voxel_size.npy'):
                raise FileNotFoundError("Required parameter files 'min_coords.npy' or 'voxel_size.npy' not found.")
            # 加载并缓存参数
            float_to_int_xyz.min_coords = np.load('min_coords.npy')
            float_to_int_xyz.voxel_size = np.load('voxel_size.npy')
        min_coords = float_to_int_xyz.min_coords
        voxel_size = float_to_int_xyz.voxel_size

        # 计算整数坐标
        Vc = points_float - min_coords
        points_int = np.floor(Vc / voxel_size).astype(np.uint64)

        return points_int
    except Exception as e:
        print(f"An error occurred in float_to_int_xyz: {e}")
        raise

def int_to_float_xyz(points_int):
    try:
        if not hasattr(int_to_float_xyz, 'min_coords'):
            # 检查文件是否存在
            if not os.path.exists('min_coords.npy') or not os.path.exists('voxel_size.npy'):
                raise FileNotFoundError("Required parameter files 'min_coords.npy' or 'voxel_size.npy' not found.")
            # 加载并缓存参数
            int_to_float_xyz.min_coords = np.load('min_coords.npy')
            int_to_float_xyz.voxel_size = np.load('voxel_size.npy')
        min_coords = int_to_float_xyz.min_coords
        voxel_size = int_to_float_xyz.voxel_size

        # 计算浮点坐标
        points_float = (points_int.astype(np.float64) + 0.5) * voxel_size + min_coords

        return points_float
    except Exception as e:
        print(f"An error occurred in int_to_float_xyz: {e}")
        raise


def install(package_path):
    # Convert to absolute path
    package_abs_path = os.path.abspath(package_path)
    
    # Debug: Check if the path exists
    if not os.path.exists(package_abs_path):
        print(f"DEBUG: The path {package_abs_path} does not exist.")
        return
    print(f"DEBUG: Installing package from {package_abs_path}")
    
    # Install the package using pip
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_abs_path])
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install package from {package_abs_path}")
        print(f"DEBUG: Subprocess error details: {e}")
        raise


def run_training(script_path, dataset_name, model_folder, iterations):
    print(f"DEBUG: Running training for {model_folder} in {script_path}")
    
    # Step 1: Run train.py
    train_command = [
        sys.executable, "train.py",
        "-s", f"../../colmap_dataset/{dataset_name}",
        "-m", f"../../retrain_model/{model_folder}",
        "--data_device", "cuda",
        "--iterations", str(iterations),
        "--eval"
    ]
    try:
        subprocess.check_call(train_command, cwd=script_path)
        print(f"Training completed for {model_folder}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed for {model_folder}")
        print(f"DEBUG: Subprocess error details: {e}")
        raise

    # Step 2: Run render.py
    render_command = [
        sys.executable, "render.py",
        "-m", f"../../retrain_model/{model_folder}",
        "-s", f"../../colmap_dataset/{dataset_name}"
    ]
    try:
        subprocess.check_call(render_command, cwd=script_path)
        print(f"Rendering completed for {model_folder}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Rendering failed for {model_folder}")
        print(f"DEBUG: Subprocess error details: {e}")
        raise

    # Step 3: Run metrics.py
    metrics_command = [
        sys.executable, "metrics.py",
        "-m", f"../../retrain_model/{model_folder}"
    ]
    try:
        subprocess.check_call(metrics_command, cwd=script_path)
        print(f"Metrics calculation completed for {model_folder}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Metrics calculation failed for {model_folder}")
        print(f"DEBUG: Subprocess error details: {e}")
        raise



def run_compression(script_path, dataset_name, model_folder, output_vq_folder):
    print(f"DEBUG: Running compression for {model_folder} in {script_path} with output to {output_vq_folder}")
    
    # Step 1: Run compress.py
    compress_command = [
        sys.executable, "compress.py",
        "--source_path", f"../../colmap_dataset/{dataset_name}",
        "--model_path", f"../../retrain_model/{model_folder}",
        "--output_vq", f"../../VQ_model/{output_vq_folder}",
        "--data_device", "cuda",
        #"--finetune_iterations", "2000",
        "--eval"
    ]
    try:
        subprocess.check_call(compress_command, cwd=script_path)
        print(f"Compression completed for {model_folder}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Compression failed for {model_folder}")
        print(f"DEBUG: Subprocess error details: {e}")
        raise

    # Step 2: Run render.py
    render_command = [
        sys.executable, "render.py",
        "-m", f"../../VQ_model/{output_vq_folder}",
        "-s", f"../../colmap_dataset/{dataset_name}"
    ]
    try:
        subprocess.check_call(render_command, cwd=script_path)
        print(f"Rendering completed for {output_vq_folder}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Rendering failed for {output_vq_folder}")
        print(f"DEBUG: Subprocess error details: {e}")
        raise

    # Step 3: Run metrics.py
    metrics_command = [
        sys.executable, "metrics.py",
        "-m", f"../../VQ_model/{output_vq_folder}"
    ]
    try:
        subprocess.check_call(metrics_command, cwd=script_path)
        print(f"Metrics calculation completed for {output_vq_folder}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Metrics calculation failed for {output_vq_folder}")
        print(f"DEBUG: Subprocess error details: {e}")
        raise

    # Step 4: Run npz2ply.py to convert the largest iteration point_cloud.npz to point_cloud.ply
    try:
        # Find the point_cloud path
        point_cloud_path = os.path.join('../../VQ_model', output_vq_folder, 'point_cloud')
        abs_point_cloud_path = os.path.abspath(os.path.join(script_path, point_cloud_path))

        # List iteration folders
        iteration_folders = []
        for folder_name in os.listdir(abs_point_cloud_path):
            if folder_name.startswith('iteration_'):
                try:
                    iteration_number = int(folder_name[len('iteration_'):])
                    iteration_folders.append((iteration_number, folder_name))
                except ValueError:
                    pass  # Ignore invalid folders

        if not iteration_folders:
            print(f"ERROR: No iteration folders found in {abs_point_cloud_path}")
            raise FileNotFoundError(f"No iteration folders found in {abs_point_cloud_path}")

        # Find the folder with the maximum iteration number
        max_iteration_number, max_iteration_folder = max(iteration_folders)

        # Build input_npz_path and output_ply_path
        input_npz_path = os.path.join(abs_point_cloud_path, max_iteration_folder, 'point_cloud.npz')
        output_ply_path = os.path.join(abs_point_cloud_path, max_iteration_folder, 'point_cloud.ply')

        # Build the command
        npz2ply_command = [
            sys.executable, 'npz2ply.py',
            input_npz_path,
            '--ply_file', output_ply_path
        ]

        # Run the command
        subprocess.check_call(npz2ply_command, cwd=script_path)
        print(f"Conversion of point_cloud.npz to point_cloud.ply completed for iteration {max_iteration_number}")
    except Exception as e:
        print(f"ERROR: Conversion of point_cloud.npz to point_cloud.ply failed for {output_vq_folder}")
        print(f"DEBUG: Error details: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voxelize point clouds with different depths')
    parser.add_argument('--depth_range', type=int, nargs=2, required=True, help='Range of depths to voxelize')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save output PLY files')
    parser.add_argument('--iterations', type=int, default=15000, help='Number of iterations for training')  # Add iterations argument
    args = parser.parse_args()

    # Construct input_ply_path based on the dataset_name
    input_ply_path = os.path.join('C:\\Users\\jay\\Desktop\\Jay_pipeline\\original_model', args.dataset_name, 'point_cloud', 'iteration_30000', 'point_cloud.ply')

    # Check if the input PLY file exists
    if not os.path.exists(input_ply_path):
        raise FileNotFoundError(f"Input PLY file not found at path: {input_ply_path}")

    # Find the dataset folder in original_model
    original_model_path = os.path.join('..', 'original_model', args.dataset_name)
    retrain_model_path = os.path.join('..', 'retrain_model')

    if not os.path.exists(original_model_path):
        raise FileNotFoundError(f"Dataset folder not found: {original_model_path}")
    
    # Copy dataset folder to retrain_model with depth suffix
    for depth in range(args.depth_range[0], args.depth_range[1] + 1):
        new_folder_name = f"{args.dataset_name}_depth_{depth}"
        new_folder_path = os.path.join(retrain_model_path, new_folder_name)
        
        if os.path.exists(new_folder_path):
            shutil.rmtree(new_folder_path)
        shutil.copytree(original_model_path, new_folder_path)

        # Delete point_cloud folder and input.ply
        point_cloud_path = os.path.join(new_folder_path, 'point_cloud')
        if os.path.exists(point_cloud_path):
            shutil.rmtree(point_cloud_path)

        input_ply_path_retrain = os.path.join(new_folder_path, 'input.ply')
        if os.path.exists(input_ply_path_retrain):
            os.remove(input_ply_path_retrain)

        # Continue with voxelization process
        PC = os.path.join(args.output_path, f"{args.dataset_name}_depth_{depth}.ply")
        PC_voxelized = os.path.join(args.output_path, f"{args.dataset_name}_depth_{depth}_voxelized.ply")
        
        voxelize_and_recoloring(depth, input_ply_path, PC, PC_voxelized, args.output_path, args.dataset_name)

    # After all processing is complete, move generated PLY files to retrain_model and rename them to input.ply
    for ply_file in os.listdir(args.output_path):
        if ply_file.endswith('.ply'):
            # Extract dataset name and depth from the filename
            dataset_name_depth = os.path.splitext(ply_file)[0]  # Remove .ply extension

            # Find the corresponding folder in retrain_model
            corresponding_folder = os.path.join(retrain_model_path, dataset_name_depth)
            
            if os.path.exists(corresponding_folder):
                # Move the PLY file to the corresponding folder and rename it to input.ply
                src_ply_file = os.path.join(args.output_path, ply_file)
                dst_ply_file = os.path.join(corresponding_folder, 'input.ply')
                shutil.move(src_ply_file, dst_ply_file)

    # Continue with the rest of the script


    # Step 1: Change directory to retrain_script and install diff-gaussian-rasterization
    retrain_script_path = 'C:\\Users\\jay\\Desktop\\Jay_pipeline\\code_GPCC\\retrain_script'
    submodule_retrain_path = 'C:\\Users\\jay\\Desktop\\Jay_pipeline\\code_GPCC\\retrain_script\\submodules\\diff-gaussian-rasterization'
    install(submodule_retrain_path)  # Install the package for retraining

    # Step 2: Traverse through depths and run retraining
    for depth in range(args.depth_range[0], args.depth_range[1] + 1):
        model_folder_name = f"{args.dataset_name}_depth_{depth}"
        model_folder_path = os.path.join(retrain_model_path, model_folder_name)
        
        if os.path.isdir(model_folder_path):
            print(f"Found model folder: {model_folder_name}")
            run_training(retrain_script_path, args.dataset_name, model_folder_name, args.iterations)
        else:
            print(f"Model folder not found: {model_folder_name}")

    # # Step 3: Change directory to VQ_script and install diff-gaussian-rasterization for compression
    # VQ_script_path = 'C:\\Users\\jay\\Desktop\\Jay_pipeline\\code\\VQ_script'
    # submodule_VQ_path = 'C:\\Users\\jay\\Desktop\\Jay_pipeline\\code\\VQ_script\\submodules\\diff-gaussian-rasterization'
    # install(submodule_VQ_path)  # Install the package for compression

    # # Step 4: Run compression for each folder in retrain_model and save output in VQ_model
    # for model_folder in os.listdir(retrain_model_path):
    #     if os.path.isdir(os.path.join(retrain_model_path, model_folder)):
    #         dataset_name = model_folder.split('_depth_')[0]
    #         output_vq_folder = model_folder
    #         run_compression(VQ_script_path, dataset_name, model_folder, output_vq_folder)




