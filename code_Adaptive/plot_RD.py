import os
import sys
import json
import re
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Decompress, reorder, and process PLY files with given pq combinations.'
    )
    parser.add_argument('--depth_start', type=int, required=True, help='Initial depth for voxelization')
    parser.add_argument('--voxel_thr', type=int, required=True, help='Threshold for small voxel point count')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--retrain_mode', type=str, required=True,
                        help='Set to "PC" for retrain PC, "3DGS" for retrain 3DGS')
    parser.add_argument('--use_adaptive', type=str, required=True,
                        help='Set to "true" for adaptive voxelization, "false" for uniform voxelization')
    parser.add_argument('--comp_mode', type=str, required=True,
                        help='Set to "lossy" for lossy covariance compression, "lossless" for lossless covariance compression')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    depth = args.depth_start
    thr = args.voxel_thr
    # Convert retrain_mode to uppercase for consistency, e.g., "PC" or "3DGS"
    mode = args.retrain_mode.upper()
    suffix = "adapt" if args.use_adaptive.lower() == "true" else "uniform"
    comp_mode = args.comp_mode.lower()

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = base_path = os.path.abspath(os.path.join(current_script_dir, '..'))

    # Select the directory of postprocess.py based on comp_mode
    if comp_mode == "lossless":
        postprocess_path = os.path.join(root_path, 'code_Adaptive', 'Lossless_covar')
    elif comp_mode == "lossy":
        postprocess_path = os.path.join(root_path, 'code_Adaptive', 'Lossy_covar')
    else:
        raise ValueError("comp_mode must be either 'lossy' or 'lossless'")
    
    # Construct the command-line arguments for postprocess.py (pass through unchanged)
    postprocess_cmd = [
        sys.executable, "postprocess.py",
        "--depth_start", str(depth),
        "--voxel_thr", str(thr),
        "--dataset_name", dataset_name,
        "--retrain_mode", mode,
        "--use_adaptive", args.use_adaptive 
    ]
    print(f"[DEBUG] Running postprocess command in {postprocess_path}: {' '.join(postprocess_cmd)}")
    # subprocess.run(postprocess_cmd, cwd=postprocess_path, check=True)

    extract_all_pq_cmd = [
        sys.executable, "extract_all_pq.py",
        "--depth_start", str(depth),
        "--voxel_thr", str(thr),
        "--dataset_name", dataset_name,
        "--retrain_mode", mode,
        "--use_adaptive", args.use_adaptive
    ]
    print(f"[DEBUG] Running postprocess command in {postprocess_path}: {' '.join(postprocess_cmd)}")
    subprocess.run(extract_all_pq_cmd, cwd=postprocess_path, check=True)



if __name__ == "__main__":
    main()

