import os
import argparse
import subprocess
import concurrent.futures
import json
from plyfile import PlyData, PlyElement
import numpy as np

# def adaptive_normalize(values, dtype):
#     min_value = np.min(values)
#     max_value = np.max(values)
#     normalized_values = ((values - min_value) / (max_value - min_value) * (np.iinfo(dtype).max)).astype(dtype)
#     return normalized_values, min_value, max_value


def adaptive_normalize(values, dtype):
    # If NaN exists or all values are identical (e.g., all 0), return a zero array
    if np.any(np.isnan(values)) or np.all(values == values[0]):
        normalized_values = np.zeros_like(values, dtype=dtype)
        return normalized_values, 0, 0

    min_value = np.min(values)
    max_value = np.max(values)
    # If denominator is 0 (i.e., all values are identical), also return a zero array
    if max_value - min_value == 0:
        normalized_values = np.zeros_like(values, dtype=dtype)
        return normalized_values, min_value, max_value

    normalized_values = ((values - min_value) / (max_value - min_value) * (np.iinfo(dtype).max)).astype(dtype)
    return normalized_values, min_value, max_value




# Function to convert NumPy types to native Python types for JSON serialization
def numpy_to_native(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    else:
        return data

# RGB to YUV conversion
def convert_rgb2yuv(rgb):
    rgb2yuv = np.array([[0.212600, -0.114572, 0.50000],
                        [0.715200, -0.385428, -0.454153],
                        [0.072200, 0.50000, -0.045847]])
    yuv = np.dot(rgb, rgb2yuv)
    return yuv

def process_ply_file(
    ply_file, point_cloud_voxelized_file, output_base_folder, 
    meta_data_output_folder, dataset_name, depth, thr, mode, suffix
):
    # Create output folder
    output_folder = os.path.join(output_base_folder, f'{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless')
    os.makedirs(output_folder, exist_ok=True)

    # Create compression folders
    os.makedirs(os.path.join(output_folder, "opacity_compressed"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "dc_compressed"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "rest_compressed"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "scale_compressed"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "rot_compressed"), exist_ok=True)

    # Load PLY file
    plydata_rest = PlyData.read(ply_file)
    plydata_voxelized = PlyData.read(point_cloud_voxelized_file)

    # Verify consistency in vertex count
    if len(plydata_rest['vertex']) != len(plydata_voxelized['vertex']):
        raise ValueError(f"Error: Vertex count mismatch between {ply_file} and {point_cloud_voxelized_file}!")

    # Extract x, y, z
    x = plydata_voxelized['vertex']['x']
    y = plydata_voxelized['vertex']['y']
    z = plydata_voxelized['vertex']['z']

    # 	Initialize metadata
    meta_data = {"Geometry": {}, "Attribute": {}}

    # # Read voxelized geometry information
    # min_coords = np.load(f'{voxelization_base_dir}/min_coords_{dataset_name}_depth_{depth}.npy').tolist()
    # voxel_size = np.load(f'{voxelization_base_dir}/voxel_size_{dataset_name}_depth_{depth}.npy').tolist()

    # meta_data["Geometry"] = {
    #     "min_coords": min_coords,
    #     "voxel_size": voxel_size
    # }

    ### --- Opacity attribute processing --- ###
    opacity, opacity_min, opacity_max = adaptive_normalize(plydata_rest['vertex']['opacity'], np.uint16)
    vertices_opacity = np.array(
        list(zip(x, y, z, opacity)), 
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('reflectance', 'u2')]
    )
    output_opacity_file = os.path.join(output_folder, f'{dataset_name}_opacity.ply')
    PlyData([PlyElement.describe(vertices_opacity, 'vertex')], text=False).write(output_opacity_file)

    meta_data["Attribute"]["opacity"] = {"min": numpy_to_native(opacity_min), "max": numpy_to_native(opacity_max)}

    ### --- DC attribute processing --- ###
    f_dc_0 = plydata_rest['vertex']['f_dc_0']
    f_dc_1 = plydata_rest['vertex']['f_dc_1']
    f_dc_2 = plydata_rest['vertex']['f_dc_2']

    rgb_dc = np.vstack((f_dc_0, f_dc_1, f_dc_2)).T
    yuv_dc = convert_rgb2yuv(rgb_dc)

    f_dc_0_norm, f_dc_0_min, f_dc_0_max = adaptive_normalize(yuv_dc[:, 0], np.uint8)
    f_dc_1_norm, f_dc_1_min, f_dc_1_max = adaptive_normalize(yuv_dc[:, 1], np.uint8)
    f_dc_2_norm, f_dc_2_min, f_dc_2_max = adaptive_normalize(yuv_dc[:, 2], np.uint8)

    meta_data["Attribute"].update({
        "f_dc_0": {"min": f_dc_0_min, "max": f_dc_0_max},
        "f_dc_1": {"min": f_dc_1_min, "max": f_dc_1_max},
        "f_dc_2": {"min": f_dc_2_min, "max": f_dc_2_max}
    })

    vertices_dc = np.array(
        list(zip(x, y, z, f_dc_0_norm, f_dc_1_norm, f_dc_2_norm)),
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    output_dc_file = os.path.join(output_folder, f'{dataset_name}_dc.ply')
    PlyData([PlyElement.describe(vertices_dc, 'vertex')], text=False).write(output_dc_file)

    ### --- Rest attribute processing --- ###
    for i in range(0, 45, 3):
        f_rest_0 = plydata_rest['vertex'][f'f_rest_{i}']
        f_rest_1 = plydata_rest['vertex'][f'f_rest_{i+1}']
        f_rest_2 = plydata_rest['vertex'][f'f_rest_{i+2}']

        rgb_rest = np.vstack((f_rest_0, f_rest_1, f_rest_2)).T
        yuv_rest = convert_rgb2yuv(rgb_rest)

        f_rest_0_norm, f_rest_0_min, f_rest_0_max = adaptive_normalize(yuv_rest[:, 0], np.uint8)
        f_rest_1_norm, f_rest_1_min, f_rest_1_max = adaptive_normalize(yuv_rest[:, 1], np.uint8)
        f_rest_2_norm, f_rest_2_min, f_rest_2_max = adaptive_normalize(yuv_rest[:, 2], np.uint8)

        meta_data["Attribute"].update({
            f"f_rest_{i}": {"min": f_rest_0_min, "max": f_rest_0_max},
            f"f_rest_{i+1}": {"min": f_rest_1_min, "max": f_rest_1_max},
            f"f_rest_{i+2}": {"min": f_rest_2_min, "max": f_rest_2_max}
        })

        vertices_rest = np.array(
            list(zip(x, y, z, f_rest_0_norm, f_rest_1_norm, f_rest_2_norm)),
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        output_rest_file = os.path.join(output_folder, f'{dataset_name}_rest_{i}_{i+1}_{i+2}.ply')
        PlyData([PlyElement.describe(vertices_rest, 'vertex')], text=False).write(output_rest_file)

        # Process scale attributes
    for i, scale_attr in enumerate(['scale_0', 'scale_1', 'scale_2']):
        scale_values = plydata_rest['vertex'][scale_attr]
        scale_norm, scale_min, scale_max = adaptive_normalize(scale_values, np.uint16)

        vertices_scale = np.array(
            list(zip(x, y, z, scale_norm)), 
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('reflectance', 'u2')]
        )
        output_scale_file = os.path.join(output_folder, f'{dataset_name}_{scale_attr}.ply')
        PlyData([PlyElement.describe(vertices_scale, 'vertex')], text=False).write(output_scale_file)

        meta_data["Attribute"][scale_attr] = {"min": numpy_to_native(scale_min), "max": numpy_to_native(scale_max)}

    # 	Process rot attributes
    for i, rot_attr in enumerate(['rot_0', 'rot_1', 'rot_2', 'rot_3']):
        rot_values = plydata_rest['vertex'][rot_attr]
        rot_norm, rot_min, rot_max = adaptive_normalize(rot_values, np.uint16)

        vertices_rot = np.array(
            list(zip(x, y, z, rot_norm)), 
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('reflectance', 'u2')]
        )
        output_rot_file = os.path.join(output_folder, f'{dataset_name}_{rot_attr}.ply')
        PlyData([PlyElement.describe(vertices_rot, 'vertex')], text=False).write(output_rot_file)

        meta_data["Attribute"][rot_attr] = {"min": numpy_to_native(rot_min), "max": numpy_to_native(rot_max)}

    # Save metadata
    meta_data_output_path = os.path.join(meta_data_output_folder, f"meta_data_{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless.json")
    os.makedirs(os.path.dirname(meta_data_output_path), exist_ok=True)
    with open(meta_data_output_path, 'w') as json_file:
        json.dump(meta_data, json_file, indent=4)

    print(f"Meta-data file saved to {meta_data_output_path}")
    return output_folder


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

# Function to find the voxelized PLY file ||train_depth_15_thr_30_voxel_xyz_adapt
def find_voxelized_ply_file(voxelization_base_dir, dataset_name, depth, thr, suffix):
    voxelized_ply_file = os.path.join(voxelization_base_dir, f"{dataset_name}_depth_{depth}_thr_{thr}_voxel_xyz_{suffix}.ply")
    if not os.path.exists(voxelized_ply_file):
        raise FileNotFoundError(f"Voxelized PLY file not found: {voxelized_ply_file}")
    print(f"Using voxelized PLY file: {voxelized_ply_file}")
    return voxelized_ply_file

# Main function
def main():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    print(f"[DEBUG] Root path determined as: {root_path}")

    retrain_model_dir = os.path.join(root_path, "retrain_model")
    voxelization_base_dir = os.path.join(root_path, "voxelized_adapt")
    output_base_folder = os.path.join(root_path, "attributes_compressed")
    meta_data_output_folder = os.path.join(root_path, "RDO", "Meta_data")
    tmc3_path = os.path.join(root_path,  "mpeg-pcc-tmc13-master", "build", "tmc3", "Release", "tmc3.exe")

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
    # Convert retrain_mode to uppercase, e.g., "PC" or "3DGS"
    mode = args.retrain_mode.upper()
    suffix = "adapt" if args.use_adaptive.lower() == "true" else "uniform"



    ply_file = find_largest_iteration(retrain_model_dir, dataset_name, depth, thr, mode, suffix)
    point_cloud_voxelized_file = find_voxelized_ply_file(voxelization_base_dir, dataset_name, depth, thr, suffix)
    
    # Get output_folder from process_ply_file
    output_folder = process_ply_file(ply_file, point_cloud_voxelized_file, output_base_folder, meta_data_output_folder, dataset_name, depth, thr, mode, suffix)

    # Define pq values for different files
    # pq_opacity = [4, 16, 28, 34, 40]
    # pq_dc = [4, 16, 20, 24, 28]
    # pq_rest = [40, 38, 34, 31, 28, 16, 4]


    pq_opacity = [4]
    pq_dc = [4]
    pq_rest = [4]
  
    

    

    # Compression process
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        # Handle opacity compression
        opacity_files = [f for f in os.listdir(output_folder) if f.endswith("_opacity.ply")]
        for pq in pq_opacity:
            for file in opacity_files:
                input_file = os.path.join(output_folder, file)
                compressed_file = os.path.join(output_folder, "opacity_compressed", f"{os.path.splitext(file)[0]}_pq_{pq}.bin")
                tmc3_opacity_compress_command = [
                    tmc3_path, f"--uncompressedDataPath={input_file}", f"--compressedStreamPath={compressed_file}",
                    "--mode=0", "--geomTreeType=0", "--partitionMethod=3", f"--partitionOctreeDepth={depth}",
                    "--convertPlyColourspace=1", "--transformType=0", "--rahtExtension=0",
                    "--rahtPredictionEnabled=0", f"--qp={pq}", "--bitdepth=8", "--colourMatrix=2",
                    "--attrOffset=0", "--attrScale=257", "--attrInterPredSearchRange=-1", "--attribute=reflectance",
                ]
                print(f"Compressing {input_file} with pq={pq}, attribute=reflectance")
                futures.append(executor.submit(subprocess.run, tmc3_opacity_compress_command))

        # Handle dc compression
        dc_files = [f for f in os.listdir(output_folder) if f.endswith("_dc.ply")]
        for pq in pq_dc:
            for file in dc_files:
                input_file = os.path.join(output_folder, file)
                compressed_file = os.path.join(output_folder, "dc_compressed", f"{os.path.splitext(file)[0]}_pq_{pq}.bin")
                tmc3_dc_compress_command = [
                    tmc3_path, f"--uncompressedDataPath={input_file}", f"--compressedStreamPath={compressed_file}",
                    "--mode=0", "--geomTreeType=0", "--partitionMethod=3", f"--partitionOctreeDepth={depth}",
                    "--convertPlyColourspace=0", "--transformType=0", "--rahtExtension=0",
                    "--rahtPredictionEnabled=0", f"--qp={pq}", "--bitdepth=8", "--colourMatrix=0",
                    "--attrOffset=0", "--attrScale=1", "--attrInterPredSearchRange=-1", "--attribute=color",  
                ]
                print(f"Compressing {input_file} with pq={pq}, attribute=color")
                futures.append(executor.submit(subprocess.run, tmc3_dc_compress_command))

        # Handle rest compression
        rest_files = [f for f in os.listdir(output_folder) if "rest_" in f and f.endswith(".ply")]
        for pq in pq_rest:
            for file in rest_files:
                input_file = os.path.join(output_folder, file)
                compressed_file = os.path.join(output_folder, "rest_compressed", f"{os.path.splitext(file)[0]}_pq_{pq}.bin")
                tmc3_rest_compress_command = [
                    tmc3_path, f"--uncompressedDataPath={input_file}", f"--compressedStreamPath={compressed_file}",
                    "--mode=0", "--geomTreeType=0", "--partitionMethod=3", f"--partitionOctreeDepth={depth}",
                    "--convertPlyColourspace=0", "--transformType=0", "--rahtExtension=0",
                    "--rahtPredictionEnabled=0", f"--qp={pq}", "--bitdepth=8", "--colourMatrix=0",
                    "--attrOffset=0", "--attrScale=1", "--attrInterPredSearchRange=-1", "--attribute=color",  
                ]
                print(f"Compressing {input_file} with pq={pq}, attribute=color")
                futures.append(executor.submit(subprocess.run, tmc3_rest_compress_command))


        # ---- Modify compression command for scale files ----
        scale_files = [f for f in os.listdir(output_folder) if f.startswith(dataset_name) and "scale_" in f and f.endswith(".ply")]
        for file in scale_files:
            input_file = os.path.join(output_folder, file)
            compressed_file = os.path.join(output_folder, "scale_compressed", f"{os.path.splitext(file)[0]}_comp.bin")
            tmc3_scale_command = [
                tmc3_path, f"--uncompressedDataPath={input_file}", f"--compressedStreamPath={compressed_file}",
                "--mode=0", "--mergeDuplicatedPoints=0", "--positionQuantizationScale=1", "--trisoupNodeSizeLog2=0",
                "--neighbourAvailBoundaryLog2=8", "--intra_pred_max_node_size_log2=6", "--inferredDirectCodingMode=1",
                "--maxNumQtBtBeforeOt=4", "--minQtbtSizeLog2=0", "--planarEnabled=1", "--planarModeIdcmUse=0",
                "--convertPlyColourspace=0", "--transformType=1", "--numberOfNearestNeighborsInPrediction=3",
                "--intraLodPredictionSkipLayers=0", "--interComponentPredictionEnabled=0",
                "--adaptivePredictionThreshold=64", "--qp=4", "--bitdepth=16", "--attrOffset=0", "--attrScale=1",
                "--attribute=reflectance"
            ]
            print(f"Compressing {input_file} (scale)")
            futures.append(executor.submit(subprocess.run, tmc3_scale_command))

        # ---- Modify compression command for rot files ----
        rot_files = [f for f in os.listdir(output_folder) if f.startswith(dataset_name) and "rot_" in f and f.endswith(".ply")]
        for file in rot_files:
            input_file = os.path.join(output_folder, file)
            compressed_file = os.path.join(output_folder, "rot_compressed", f"{os.path.splitext(file)[0]}_comp.bin")
            tmc3_rot_command = [
                tmc3_path, f"--uncompressedDataPath={input_file}", f"--compressedStreamPath={compressed_file}",
                "--mode=0", "--mergeDuplicatedPoints=0", "--positionQuantizationScale=1", "--trisoupNodeSizeLog2=0",
                "--neighbourAvailBoundaryLog2=8", "--intra_pred_max_node_size_log2=6", "--inferredDirectCodingMode=1",
                "--maxNumQtBtBeforeOt=4", "--minQtbtSizeLog2=0", "--planarEnabled=1", "--planarModeIdcmUse=0",
                "--convertPlyColourspace=0", "--transformType=1", "--numberOfNearestNeighborsInPrediction=3",
                "--intraLodPredictionSkipLayers=0", "--interComponentPredictionEnabled=0",
                "--adaptivePredictionThreshold=64", "--qp=4", "--bitdepth=16", "--attrOffset=0", "--attrScale=1",
                "--attribute=reflectance"
            ]
            print(f"Compressing {input_file} (rot)")
            futures.append(executor.submit(subprocess.run, tmc3_rot_command))


        concurrent.futures.wait(futures)
        print("All compression tasks are done!")

if __name__ == "__main__":
    main()
