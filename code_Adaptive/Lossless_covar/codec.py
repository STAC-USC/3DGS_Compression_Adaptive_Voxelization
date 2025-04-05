import os
import sys
import argparse
import subprocess
import shutil

def main():
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
    mode = args.retrain_mode.upper()
    suffix = "adapt" if args.use_adaptive.lower() == "true" else "uniform"


    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    print(f"[DEBUG] Root path determined as: {root_path}")

    retrain_model_dir = os.path.join(root_path, "retrain_model")
    vq_model_dir = os.path.join(root_path, "VQ_model")
    colmap_dataset_dir = os.path.join(root_path, "colmap_dataset")
    script_path = os.path.join(root_path, "code_Adaptive", "VQ_script")

    suffix = "adapt" if args.use_adaptive.lower() == "true" else "uniform"
    mode = args.retrain_mode.upper()
    model_folder_name = f"{args.dataset_name}_depth_{args.depth_start}_thr_{args.voxel_thr}_{mode}_{suffix}"
    model_folder = os.path.join(retrain_model_dir, model_folder_name)
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    print(f"[DEBUG] Found model folder: {model_folder}")



    source_path = os.path.join(colmap_dataset_dir, args.dataset_name)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Colmap dataset folder not found: {source_path}")

    # 打印调试信息
    print(f"DEBUG: Running compression for model folder: {model_folder}")
    print(f"DEBUG: Using script path: {script_path}")
    print(f"DEBUG: Source dataset path: {source_path}")
  


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
