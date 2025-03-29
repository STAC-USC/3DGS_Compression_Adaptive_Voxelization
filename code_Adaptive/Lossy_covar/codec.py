import os
import sys
import argparse
import subprocess
import shutil

def run_compression(script_path, dataset_name, model_folder, output_vq_folder, colmap_dataset_path):
    print(f"DEBUG: Running compression for {model_folder} in {script_path} with output to {output_vq_folder}")
    
    # Step 1: Run compress.py
    source_path = os.path.join(colmap_dataset_path, dataset_name)
    compress_command = [
        "python", "compress.py",
        "--source_path", source_path,
        "--model_path", model_folder,
        "--output_vq", output_vq_folder,
        # Set if you want to quickly test
        # "--finetune_iterations", "1000", 
        "--data_device", "cuda",
        "--eval"
    ]
    try:
        subprocess.check_call(compress_command, cwd=script_path)
        print(f"Compression completed for {model_folder}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Compression failed for {model_folder}")
        print(f"DEBUG: Subprocess error details: {e}")
        return

    # # Step 2: Run render.py
    # render_command = [
    #     "python", "render.py",
    #     "-m", output_vq_folder,
    #     "-s", source_path,
    # ]
    # try:
    #     subprocess.check_call(render_command, cwd=script_path)
    #     print(f"Rendering completed for {output_vq_folder}")
    # except subprocess.CalledProcessError as e:
    #     print(f"ERROR: Rendering failed for {output_vq_folder}")
    #     print(f"DEBUG: Subprocess error details: {e}")
    #     raise

    # # Step 3: Run metrics.py
    # metrics_command = [
    #     "python", "metrics.py",
    #     "-m", output_vq_folder
    # ]
    # try:
    #     subprocess.check_call(metrics_command, cwd=script_path)
    #     print(f"Metrics calculation completed for {output_vq_folder}")
    # except subprocess.CalledProcessError as e:
    #     print(f"ERROR: Metrics calculation failed for {output_vq_folder}")
    #     print(f"DEBUG: Subprocess error details: {e}")
    #     raise

    # Step 4: Run npz2ply.py to convert the largest iteration point_cloud.npz to point_cloud.ply
    try:
        point_cloud_path = os.path.join(output_vq_folder, 'point_cloud')
        abs_point_cloud_path = os.path.abspath(point_cloud_path)

        # List iteration folders
        iteration_folders = []
        for folder_name in os.listdir(abs_point_cloud_path):
            if folder_name.startswith('iteration_'):
                try:
                    iteration_number = int(folder_name[len('iteration_'):])
                    iteration_folders.append((iteration_number, folder_name))
                except ValueError:
                    pass  # Ignore invalid folder names

        if not iteration_folders:
            print(f"ERROR: No iteration folders found in {abs_point_cloud_path}")
            raise FileNotFoundError(f"No iteration folders found in {abs_point_cloud_path}")

        # Find the folder corresponding to the maximum number of iterations
        max_iteration_number, max_iteration_folder = max(iteration_folders)

        # Construct the input and output paths for the npz to ply conversion
        input_npz_path = os.path.join(abs_point_cloud_path, max_iteration_folder, 'point_cloud.npz')
        output_ply_path = os.path.join(abs_point_cloud_path, max_iteration_folder, 'point_cloud.ply')

        npz2ply_command = [
            sys.executable, 'npz2ply.py',
            input_npz_path,
            '--ply_file', output_ply_path
        ]
        subprocess.check_call(npz2ply_command, cwd=script_path)
        print(f"Conversion of point_cloud.npz to point_cloud.ply completed for iteration {max_iteration_number}")
    except Exception as e:
        print(f"ERROR: Conversion of point_cloud.npz to point_cloud.ply failed for {output_vq_folder}")
        print(f"DEBUG: Error details: {e}")
        raise

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Run compression and metrics for a given model folder')
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


    # Set the two folders before the current script directory as the root directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    print(f"[DEBUG] Root path determined as: {root_path}")

    # Define the retrain_model and VQ_model folder paths (relative to the root directory)
    retrain_model_dir = os.path.join(root_path, "retrain_model")
    vq_model_dir = os.path.join(root_path, "VQ_model")
    colmap_dataset_dir = os.path.join(root_path, "colmap_dataset")
    # Assume compress.py, render.py, metrics.py, npz2ply.py are in the directory code_YUV/VQ_script
    script_path = os.path.join(root_path, "code_Adaptive", "VQ_script")

    # Construct the target model folder name based on command line parameters
    # Determine the suffix based on --use_adaptive
    suffix = "adapt" if args.use_adaptive.lower() == "true" else "uniform"
    # retrain_mode is converted to uppercase, such as "PC" or "3DGS"
    mode = args.retrain_mode.upper()
    model_folder_name = f"{args.dataset_name}_depth_{args.depth_start}_thr_{args.voxel_thr}_{mode}_{suffix}"
    model_folder = os.path.join(retrain_model_dir, model_folder_name)
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    print(f"[DEBUG] Found model folder: {model_folder}")

    # Create a folder with the same name as output_vq under VQ_model
    output_vq_folder = os.path.join(vq_model_dir, model_folder_name)
    if not os.path.exists(output_vq_folder):
        os.makedirs(output_vq_folder)
    print(f"[DEBUG] VQ output folder: {output_vq_folder}")

    # Modify source_path to use colmap_dataset_dir
    # Here source_path will be used as the --source_path parameter of compress.py
    # Here by default there is a dataset_name folder under colmap_dataset_dir
    # For example: {root_path}/colmap_dataset/{dataset_name}
    source_path = os.path.join(colmap_dataset_dir, args.dataset_name)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Colmap dataset folder not found: {source_path}")

    # 打印调试信息
    print(f"DEBUG: Running compression for model folder: {model_folder}")
    print(f"DEBUG: Using script path: {script_path}")
    print(f"DEBUG: Source dataset path: {source_path}")
    print(f"DEBUG: Output VQ folder: {output_vq_folder}")

    # # Step 1 ~ Step 4：依次运行 compress.py, render.py, metrics.py, npz2ply.py
    # try:
    #     run_compression(script_path, args.dataset_name, model_folder, output_vq_folder, colmap_dataset_dir)
    # except Exception as e:
    #     print(f"An error occurred during compression and metrics processing: {e}")
    #     sys.exit(1)


    encoder_command_template = (
        'python encoder.py '
        '--depth_start {depth_start} '
        '--voxel_thr {voxel_thr} '
        '--dataset_name {dataset_name} '
        '--retrain_mode {retrain_mode} '
        '--use_adaptive {use_adaptive}'
    )
    decoder_command_template = (
        'python decoder.py '
        '--depth_start {depth_start} '
        '--voxel_thr {voxel_thr} '
        '--dataset_name {dataset_name} '
        '--retrain_mode {retrain_mode} '
        '--use_adaptive {use_adaptive}'
    )

    encoder_cmd = encoder_command_template.format(
        depth_start=args.depth_start,
        voxel_thr=args.voxel_thr,
        dataset_name=args.dataset_name,
        retrain_mode=args.retrain_mode,
        use_adaptive=args.use_adaptive
    )
    print(f"Running encoder command: {encoder_cmd}")
    subprocess.run(encoder_cmd, shell=True, check=True)

    decoder_cmd = decoder_command_template.format(
        depth_start=args.depth_start,
        voxel_thr=args.voxel_thr,
        dataset_name=args.dataset_name,
        retrain_mode=args.retrain_mode,
        use_adaptive=args.use_adaptive
    )
    print(f"Running decoder command: {decoder_cmd}")
    subprocess.run(decoder_cmd, shell=True, check=True)

        

if __name__ == '__main__':
    main()
