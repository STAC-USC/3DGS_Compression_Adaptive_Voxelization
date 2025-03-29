import os
import shutil
import subprocess
import sys
import glob
import argparse

def find_all_txt_files(directory, suffix):
    """
    Find and return all .txt files in the directory that match the given suffix and follow the correct PQ naming convention.
    """
    txt_files = glob.glob(os.path.join(directory, f'*{suffix}*.txt'))

    # Ensure that we're only picking up the correct files that have a single PQ suffix (extracted from bin file names)
    filtered_txt_files = [f for f in txt_files if '_pq_' in f]  # Ensure only files with a PQ value are considered
    return filtered_txt_files


def safe_rename(src, dest):
    """Safely rename a file by removing the destination if it exists."""
    if os.path.exists(dest):
        print(f"File {dest} already exists. Overwriting.")
        os.remove(dest)  # Delete existing file
    os.rename(src, dest)  # Perform rename

def process_ply_files(base_path):
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

    # Define the paths
    reconstructed_path = os.path.join(base_path, 'reconstructed_3DGS',  f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossy")
    test_model_path = os.path.join(base_path, 'test_model')
    rdo_psnr_path = os.path.join(base_path, 'RDO', 'PSNR', f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossy")
    rdo_psnr_per_view_path = os.path.join(base_path, 'RDO', 'PSNR_per_view', f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossy")
    attributes_compressed_path = os.path.join(base_path, 'attributes_compressed')
    bpp_path = os.path.join(base_path, 'RDO', 'bpp', f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossy")
    vq_model_path = os.path.join(base_path, 'VQ_model')
    retrain_script_path = os.path.join(base_path, 'code_Adaptive', 'retrain_3DGS')

    # Ensure that destination directories exist
    os.makedirs(bpp_path, exist_ok=True)
    os.makedirs(rdo_psnr_path, exist_ok=True)
    os.makedirs(rdo_psnr_per_view_path, exist_ok=True)

    # Step 1: Traverse all ply files under reconstructed_3DGS and move the corresponding log files
    ply_files = glob.glob(os.path.join(reconstructed_path, '*.ply'))

    for ply_file in ply_files:
        filename = os.path.basename(ply_file)
        dataset_name, depth = extract_dataset_name_and_depth(filename)

        attributes_folder = os.path.join(attributes_compressed_path, f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossy")

        # Process decompression logs
        for subfolder in ['dc_decompressed', 'rest_decompressed', 'opacity_decompressed']:
            subfolder_path = os.path.join(attributes_folder, subfolder)
            if not os.path.exists(subfolder_path):
                continue  # Skip if subfolder does not exist
            for root, _, files in os.walk(subfolder_path):
                for file in files:
                    if file.endswith('.txt'):
                        log_file = os.path.join(root, file)
                        dest_log_file = os.path.join(bpp_path, os.path.basename(log_file))
                        # Ensure destination directory exists
                        os.makedirs(os.path.dirname(dest_log_file), exist_ok=True)
                        shutil.move(log_file, dest_log_file)

        # Move bitstream_sizes.json
        vq_folder = os.path.join(vq_model_path, f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}")
        bitstream_json_path = os.path.join(vq_folder, 'bitstream_sizes.json')

        if os.path.exists(bitstream_json_path):
            renamed_bitstream_json = os.path.join(bpp_path, f'{dataset_name}_depth_{depth}_covariance.json')
            shutil.copy(bitstream_json_path, renamed_bitstream_json)

    install_diff_gaussian_rasterization(retrain_script_path)

    # Step 3: Process each PLY file
    for ply_file in ply_files:
        filename = os.path.basename(ply_file)
        dataset_name, depth = extract_dataset_name_and_depth(filename)

        print(f'Processing: {filename}')

        dataset_model_folder = os.path.join(test_model_path, dataset_name)
        if not os.path.exists(dataset_model_folder):
            print(f"Model folder {dataset_model_folder} not found, skipping.")
            continue

        iteration_path = os.path.join(dataset_model_folder, 'point_cloud', 'iteration_30000')
        os.makedirs(iteration_path, exist_ok=True)

        original_point_cloud_path = os.path.join(iteration_path, 'point_cloud.ply')
        if os.path.exists(original_point_cloud_path):
            os.remove(original_point_cloud_path)

        dest_ply_path = os.path.join(iteration_path, 'point_cloud.ply')
        shutil.copyfile(ply_file, dest_ply_path)

        iteration_7000_path = os.path.join(dataset_model_folder, 'point_cloud', 'iteration_7000')
        if os.path.exists(iteration_7000_path):
            shutil.rmtree(iteration_7000_path)

        # Change directory to retrain_script_path before running render.py and metrics.py
        os.chdir(retrain_script_path)

        render_command = [
            sys.executable, 'render.py', '-m', dataset_model_folder,
            '-s', os.path.join(base_path, 'colmap_dataset', dataset_name),
            '--data_device', 'cuda'
        ]
        subprocess.run(render_command, check=True)

        metrics_command = [
            sys.executable, 'metrics.py', '-m', dataset_model_folder
        ]
        subprocess.run(metrics_command, check=True)

        result_json_path = os.path.join(dataset_model_folder, 'results.json')
        per_view_json_path = os.path.join(dataset_model_folder, 'per_view.json')

        if os.path.exists(result_json_path):
            result_dest_copy = os.path.join(rdo_psnr_path, os.path.basename(result_json_path))
            shutil.copy(result_json_path, result_dest_copy)

        if os.path.exists(per_view_json_path):
            per_view_dest_copy = os.path.join(rdo_psnr_per_view_path, os.path.basename(per_view_json_path))
            shutil.copy(per_view_json_path, per_view_dest_copy)

        # Safely rename the result.json and per_view.json
        new_base_name = filename.replace('.ply', '')

        renamed_result_json = os.path.join(rdo_psnr_path, f"{new_base_name}.json")
        safe_rename(result_dest_copy, renamed_result_json)  # Use safe rename

        renamed_per_view_json = os.path.join(rdo_psnr_per_view_path, f"{new_base_name}.json")
        safe_rename(per_view_dest_copy, renamed_per_view_json)  # Use safe rename

        print(f'Completed processing {filename}')


def install_diff_gaussian_rasterization(retrain_script_path):
    # Switch to retrain_script directory
    os.chdir(retrain_script_path)

    # Install diff-gaussian-rasterization
    pip_install_command = [
        sys.executable, "-m", "pip", "install", "./submodules/diff-gaussian-rasterization"
    ]
    subprocess.run(pip_install_command, check=True)


def extract_dataset_name_and_depth(filename):
    # Extract dataset_name and depth
    parts = filename.split('_')
    dataset_name = parts[0]
    depth = parts[2]  # Assuming depth is the third part
    return dataset_name, depth


if __name__ == '__main__':
    # Base path input
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    process_ply_files(base_path)
