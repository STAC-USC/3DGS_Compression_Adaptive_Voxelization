import os
import subprocess
import argparse
import numpy as np
import json
from plyfile import PlyData, PlyElement
from concurrent.futures import ThreadPoolExecutor

# Denormalization functions using adaptive ranges from metadata
def uchar_to_float(value, min_val, max_val):
    return (value / 255.0) * (max_val - min_val) + min_val

def uint16_to_float(value, min_val, max_val):
    return (value / 65535.0) * (max_val - min_val) + min_val

# Function to read metadata from the JSON file
def load_metadata(meta_data_file):
    try:
        with open(meta_data_file, 'r') as file:
            meta_data = json.load(file)
        return meta_data
    except Exception as e:
        print(f"Error loading metadata file {meta_data_file}: {e}")
        return None

# YUV to RGB conversion function
def convert_yuv2rgb(yuv):
    yuv2rgb = np.array([[1.0, 1.0, 1.0],
                        [-2.94998246485263e-07, -0.18732418151803, 1.85559996313466],
                        [1.57479993240292, -0.468124212249768, -4.02047471671933e-07]])
    rgb = np.dot(yuv, yuv2rgb)
    return rgb

# Function to decompress a single file
def decompress_single_file(bin_file, encoded_folder, decompressed_folder, tmc3_path, log_file_base, depth):
    # Extract pq_value from bin file name (assumes bin file has _pq_X in its name)
    pq_value = bin_file.split('_pq_')[-1].replace('.bin', '')

    # Create a new log file for each individual command execution
    log_file =  f"{log_file_base}{bin_file.split('_', 1)[0]}_depth_{depth}_{bin_file.split('_', 1)[1].replace('.bin', '')}.txt"
    
    compressed_file = os.path.join(encoded_folder, bin_file)
    decompressed_file = os.path.join(decompressed_folder, bin_file.replace(".bin", ".ply"))

    tmc3_decompress_command = f"{tmc3_path} --mode=1 --outputBinaryPly=1 --compressedStreamPath=\"{compressed_file}\" --reconstructedDataPath=\"{decompressed_file}\""

    print(f"Running: {tmc3_decompress_command}")

    # Open a new log file for this command execution
    with open(log_file, 'w') as log:
        log.write(f"Running: {tmc3_decompress_command}\n")
        try:
            # Capture only the current command's stdout and stderr
            subprocess.run(tmc3_decompress_command, shell=True, check=True, stdout=log, stderr=log)
        except subprocess.CalledProcessError as e:
            log.write(f"Error during decompression: {e}\n")
            print(f"Error during decompression: {e}")

# Function to decompress multiple files using threading
def decompress_files(encoded_folder, decompressed_folder, tmc3_path, log_file_base, depth, max_threads=4):
    if not os.path.exists(decompressed_folder):
        os.makedirs(decompressed_folder)

    bin_files = [f for f in os.listdir(encoded_folder) if f.endswith(".bin")]

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit a decompression task for each bin_file
        for bin_file in bin_files:
            executor.submit(decompress_single_file, bin_file, encoded_folder, decompressed_folder, tmc3_path, log_file_base, depth)

# Function to read PLY file and extract points, colors, or reflectance
def read_ply(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None, None

    try:
        plydata = PlyData.read(file_path)
        points = np.vstack([plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']]).T

        colors = None
        reflectance = None
        
        if 'red' in plydata['vertex'].data.dtype.names and 'green' in plydata['vertex'].data.dtype.names and 'blue' in plydata['vertex'].data.dtype.names:
            colors = np.vstack([plydata['vertex'].data['red'], plydata['vertex'].data['green'], plydata['vertex'].data['blue']]).T
        elif 'refc' in plydata['vertex'].data.dtype.names:
            reflectance = plydata['vertex'].data['refc']

        print(f"Imported PLY file: {file_path}")
        return points, colors, reflectance
    except Exception as e:
        print(f"Error reading PLY file {file_path}: {e}")
        return None, None, None

# Morton order sort in-memory
def morton_order_sort(points):
    try:
        min_coords = np.min(points, axis=0)
        shifted_points = points - min_coords
        bounding_box_size = np.max(shifted_points)
        voxel_size = bounding_box_size / (np.max(shifted_points) + 1)
        voxel_indices = np.floor(shifted_points / voxel_size).astype(np.uint64)
        max_voxel_coord = np.max(voxel_indices)
        J = int(np.ceil(np.log2(max_voxel_coord + 1)))
        M = get_morton_code(voxel_indices, J)
        sort_idx = np.argsort(M)
        sorted_points = points[sort_idx]
        return sorted_points, sort_idx
    except Exception as e:
        print(f"Error in Morton order sort: {e}")
        return points, np.arange(len(points))

# Get Morton code for sorting
def get_morton_code(V, J):
    V = V.astype(np.uint64)
    N = V.shape[0]
    M = np.zeros(N, dtype=np.uint64)
    for i in range(J):
        bits = (V >> i) & 1
        M |= (bits[:, 2] << (3 * i + 0)) | (bits[:, 1] << (3 * i + 1)) | (bits[:, 0] << (3 * i + 2))
    return M

# Load template PLY file, copy processed attributes to it, and save the final output
def assign_attributes_and_save(template_ply_file, output_file, denormalized_dc, denormalized_rest, denormalized_opacity, vq_point_cloud_ply):
    try:
        # Read template point cloud
        plydata = PlyData.read(template_ply_file)
        points = plydata['vertex']  # Get all points from template

        print(f"Assigning attributes to template PLY file: {template_ply_file}")

        # Read VQ point cloud PLY
        vq_plydata = PlyData.read(vq_point_cloud_ply)
        vq_vertices = vq_plydata['vertex']

        # Ensure the number of points matches
        if len(points) != len(vq_vertices):
            raise ValueError("The number of vertices in template PLY and VQ PLY does not match.")

        print(f"Assigning attributes from VQ point cloud PLY: {vq_point_cloud_ply}")

        # Copy specified attributes
        properties_to_copy = ['scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        for prop in properties_to_copy:
            if prop in vq_vertices.data.dtype.names:
                points[prop] = vq_vertices[prop]
                print(f"Assigned {prop}.")
            else:
                print(f"Property {prop} not found in VQ point cloud PLY.")

        # Assign f_dc_* attributes
        points['f_dc_0'] = denormalized_dc[0]
        points['f_dc_1'] = denormalized_dc[1]
        points['f_dc_2'] = denormalized_dc[2]
        print(f"Assigned f_dc_0, f_dc_1, f_dc_2.")

        # Assign f_rest_* attributes
        for i in range(len(denormalized_rest)):
            points[f'f_rest_{i}'] = denormalized_rest[i]
        print(f"Assigned f_rest_0 to f_rest_{len(denormalized_rest)-1}.")

        # Assign opacity attribute
        points['opacity'] = denormalized_opacity
        print(f"Assigned opacity.")

        # Save the final PLY file
        PlyData(plydata).write(output_file)
        print(f"Final file saved: {output_file}")
    except Exception as e:
        print(f"Error assigning attributes to PLY file {output_file}: {e}")

# Function to process rest files and convert YUV to RGB
def process_rest_files(dataset_name, pq_rest, rest_folder, meta_data):
    denormalized_rest_rgb = []
    for i in range(0, 45, 3):
        rest_ply = os.path.join(rest_folder, f"{dataset_name}_rest_{i}_{i+1}_{i+2}_pq_{pq_rest}.ply")
        print(f"Processing rest PLY file: {rest_ply}")
        rest_points, rest_colors, _ = read_ply(rest_ply)
        
        sorted_rest_points, rest_sort_idx = morton_order_sort(rest_points)

        if rest_colors is not None:
            # Denormalize components
            f_rest_0 = uchar_to_float(rest_colors[rest_sort_idx][:, 0], meta_data['Attribute'][f'f_rest_{i}']['min'], meta_data['Attribute'][f'f_rest_{i}']['max'])
            f_rest_1 = uchar_to_float(rest_colors[rest_sort_idx][:, 1], meta_data['Attribute'][f'f_rest_{i+1}']['min'], meta_data['Attribute'][f'f_rest_{i+1}']['max'])
            f_rest_2 = uchar_to_float(rest_colors[rest_sort_idx][:, 2], meta_data['Attribute'][f'f_rest_{i+2}']['min'], meta_data['Attribute'][f'f_rest_{i+2}']['max'])

            # Stack YUV components and convert to RGB
            yuv_rest = np.vstack((f_rest_0, f_rest_1, f_rest_2)).T  # Shape (N, 3)
            rgb_rest = convert_yuv2rgb(yuv_rest)
            R_rest, G_rest, B_rest = rgb_rest[:, 0], rgb_rest[:, 1], rgb_rest[:, 2]

            # Append RGB components to the list
            denormalized_rest_rgb.extend([R_rest, G_rest, B_rest])

            print(f"Processed and denormalized rest_{i}_{i+1}_{i+2} for pq_{pq_rest}")
        else:
            print(f"Warning: No color data found for {rest_ply}")
        
    return denormalized_rest_rgb

# Function to process dc file
def process_dc_file(dc_ply, meta_data):
    dc_points, dc_colors, _ = read_ply(dc_ply)
    sorted_dc_points, dc_sort_idx = morton_order_sort(dc_points)
    # Denormalize components
    f_dc_0 = uchar_to_float(dc_colors[dc_sort_idx][:, 0], meta_data['Attribute']['f_dc_0']['min'], meta_data['Attribute']['f_dc_0']['max'])
    f_dc_1 = uchar_to_float(dc_colors[dc_sort_idx][:, 1], meta_data['Attribute']['f_dc_1']['min'], meta_data['Attribute']['f_dc_1']['max'])
    f_dc_2 = uchar_to_float(dc_colors[dc_sort_idx][:, 2], meta_data['Attribute']['f_dc_2']['min'], meta_data['Attribute']['f_dc_2']['max'])

    # Convert YUV to RGB
    yuv_dc = np.vstack((f_dc_0, f_dc_1, f_dc_2)).T
    rgb_dc = convert_yuv2rgb(yuv_dc)
    denormalized_dc = [rgb_dc[:, i] for i in range(3)]
    return denormalized_dc, sorted_dc_points

# Function to find the largest iteration_x folder 
def find_largest_iteration(retrain_model_dir, dataset_name, depth, thr, mode, suffix):
    dataset_dir = os.path.join(retrain_model_dir, f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}")
    point_cloud_dir = os.path.join(dataset_dir, 'point_cloud')

    # Find the largest iteration_x directory
    iteration_dirs = [d for d in os.listdir(point_cloud_dir) if d.startswith("iteration_") and os.path.isdir(os.path.join(point_cloud_dir, d))]
    if not iteration_dirs:
        raise FileNotFoundError(f"No iteration_x folders found in {point_cloud_dir}")

    # Sort by the numerical value of iteration_x
    largest_iteration_dir = max(iteration_dirs, key=lambda d: int(d.split("_")[1]))
    ply_file = os.path.join(point_cloud_dir, largest_iteration_dir, "point_cloud.ply")

    if not os.path.exists(ply_file):
        raise FileNotFoundError(f"PLY file not found in {largest_iteration_dir}")

    print(f"Using PLY file from {largest_iteration_dir}")
    return ply_file

# Function to find the VQ point cloud PLY file
def find_vq_point_cloud_ply(vq_path, dataset_name, depth, thr, mode, suffix):

    model_dir = os.path.join(vq_path, f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}")
    point_cloud_dir = os.path.join(model_dir, "point_cloud")

    # Find the largest iteration_x folder
    iteration_dirs = [d for d in os.listdir(point_cloud_dir) if d.startswith("iteration_") and os.path.isdir(os.path.join(point_cloud_dir, d))]
    if not iteration_dirs:
        raise FileNotFoundError(f"No iteration_x folders found in {point_cloud_dir}")

    # Sort by the numerical value of iteration_x
    largest_iteration_dir = max(iteration_dirs, key=lambda d: int(d.split("_")[1]))
    point_cloud_ply_path = os.path.join(point_cloud_dir, largest_iteration_dir, "point_cloud.ply")

    if not os.path.exists(point_cloud_ply_path):
        raise FileNotFoundError(f"point_cloud.ply not found in {largest_iteration_dir}")

    print(f"Using VQ point cloud PLY file from {largest_iteration_dir}")
    return point_cloud_ply_path

# Main function to decompress, reorder, and process
def main():

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    print(f"[DEBUG] Root path determined as: {root_path}")

    parser = argparse.ArgumentParser(description='Decompress, reorder, and process PLY files with given pq combinations.')
    parser.add_argument('--depth_start', type=int, required=True, help='Initial depth for voxelization')
    parser.add_argument('--voxel_thr', type=int, required=True, help='Threshold for small voxel point count')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--retrain_mode', type=str, required=True,
                        help='Set to "PC" for retrain PC, "3DGS" for retrain 3DGS')
    parser.add_argument('--use_adaptive', type=str, required=True,
                        help='Set to "true" for adaptive voxelization, "false" for uniform voxelization')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    depth = args.depth_start
    thr = args.voxel_thr
    # retrain_mode is converted to uppercase, such as "PC" or "3DGS"
    mode = args.retrain_mode.upper()
    suffix = "adapt" if args.use_adaptive.lower() == "true" else "uniform"



    output_base_dir = os.path.join(root_path, "attributes_compressed", f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossy")
    tmc3_path = os.path.join(root_path,  "mpeg-pcc-tmc13-master", "build", "tmc3", "Release", "tmc3.exe")
    ply_base_dir = os.path.join(root_path, "retrain_model")
    vq_path = os.path.join(root_path, "VQ_model") 



    # Define compressed and decompressed folder names
    compression_folders = {
        "dc": "dc_compressed",
        "opacity": "opacity_compressed",
        "rest": "rest_compressed"
    }
    decompression_folders = {
        "dc": "dc_decompressed",
        "opacity": "opacity_decompressed",
        "rest": "rest_decompressed"
    }



    # Base log file names
    log_file_bases = {
        "dc": os.path.join(output_base_dir, decompression_folders["dc"], ""),
        "opacity": os.path.join(output_base_dir, decompression_folders["opacity"], ""),
        "rest": os.path.join(output_base_dir, decompression_folders["rest"], "")
    }

    # Decompress for each attribute (rest, dc, opacity)
    decompress_files(os.path.join(output_base_dir, compression_folders["rest"]),
                     os.path.join(output_base_dir, decompression_folders["rest"]),
                     tmc3_path, log_file_bases["rest"], depth)

    decompress_files(os.path.join(output_base_dir, compression_folders["dc"]),
                     os.path.join(output_base_dir, decompression_folders["dc"]),
                     tmc3_path, log_file_bases["dc"], depth)

    decompress_files(os.path.join(output_base_dir, compression_folders["opacity"]),
                     os.path.join(output_base_dir, decompression_folders["opacity"]),
                     tmc3_path, log_file_bases["opacity"], depth)

    # Load metadata file for denormalization
    meta_data_file = os.path.abspath(os.path.join(root_path, "RDO", "Meta_data", f"meta_data_{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossy.json"))
    meta_data = load_metadata(meta_data_file)

    if meta_data is None:
        print(f"Error: Metadata file could not be loaded.")
        return

    allowed_pq_combinations = [
        # (f_rest_qp, f_dc_qp, opacity_qp)
        (40, 4, 16), (40, 4, 34), (40, 4, 40),
        (40, 16, 16), (40, 16, 34), (40, 16, 40),
        (40, 20, 16), (40, 20, 34), (40, 20, 40),
        (40, 24, 16), (40, 24, 34), (40, 24, 40),
        (40, 28, 16), (40, 28, 34), (40, 28, 40),
	    (38, 4, 4), (38, 16, 4),
        (34, 4, 4), (34, 16, 4),
        (31, 4, 4), (31, 16, 4),
        (28, 4, 4), (28, 16, 4),
        (38, 4, 16), (38, 16, 16),
        (34, 4, 16), (34, 16, 16),
        (31, 4, 16), (31, 16, 16),
        (28, 4, 16), (28, 16, 16),
        (38, 4, 28), (38, 16, 28),
        (34, 4, 28), (34, 16, 28),
        (31, 4, 28), (31, 16, 28),
        (28, 4, 28), (28, 16, 28),
        (16, 4, 4), (16, 16, 4),
        (4, 4, 4), (4, 16, 4),
        (16, 4, 16), (4, 4, 16),
      
        # (4, 4, 4)
    ]

    template_ply_file = find_largest_iteration(ply_base_dir, dataset_name, depth, thr, mode, suffix)

    # Get the VQ point cloud PLY file
    vq_point_cloud_ply =find_vq_point_cloud_ply(vq_path, dataset_name, depth, thr, mode, suffix)

    for pq_rest, pq_dc, pq_opacity in allowed_pq_combinations:
        print(f"\nProcessing combination: rest_pq_{pq_rest}, dc_pq_{pq_dc}, opacity_pq_{pq_opacity}")

        # Process rest files
        rest_folder = os.path.join(output_base_dir, decompression_folders["rest"])
        denormalized_rest = process_rest_files(dataset_name, pq_rest, rest_folder, meta_data=meta_data)

        # Process dc file
        dc_ply = os.path.join(output_base_dir, decompression_folders["dc"], f"{dataset_name}_dc_pq_{pq_dc}.ply")
        denormalized_dc, sorted_dc_points = process_dc_file(dc_ply, meta_data)

        # Process opacity file
        opacity_ply = os.path.join(output_base_dir, decompression_folders["opacity"], f"{dataset_name}_opacity_pq_{pq_opacity}.ply")
        opacity_points, _, reflectance = read_ply(opacity_ply)
        sorted_opacity_points, opacity_sort_idx = morton_order_sort(opacity_points)
        denormalized_opacity = uint16_to_float(reflectance[opacity_sort_idx], meta_data['Attribute']['opacity']['min'], meta_data['Attribute']['opacity']['max'])

        # Assign attributes and save
        output_dir = os.path.join(root_path, "reconstructed_3DGS", f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossy")
        os.makedirs(output_dir, exist_ok=True)  # 自动创建目录，如果已经存在则不会报错

        output_file = os.path.join(output_dir, f"{dataset_name}_depth_{depth}_rest_pq_{pq_rest}_dc_pq_{pq_dc}_opacity_pq_{pq_opacity}.ply")

        assign_attributes_and_save(template_ply_file, output_file, denormalized_dc, denormalized_rest, denormalized_opacity, vq_point_cloud_ply)

if __name__ == "__main__":
    main()
