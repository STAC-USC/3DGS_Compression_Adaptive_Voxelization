import os
import subprocess
import sys

def run_render_and_metrics(script_path, model_folder_path, colmap_dataset_path):
    # Extract dataset_name from model_folder's name (assuming it's before '_depth_' part)
    model_folder_name = os.path.basename(model_folder_path)
    dataset_name = model_folder_name.split('_depth_')[0]

    # Step 1: Run render.py
    render_command = [
        "python", "render.py",
        "-m", model_folder_path,
	"--data_device", "cuda",
        "-s", os.path.join(colmap_dataset_path, dataset_name)
	
    ]
    try:
        subprocess.check_call(render_command, cwd=script_path)
        print(f"Rendering completed for {model_folder_name}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Rendering failed for {model_folder_name}")
        print(f"DEBUG: Subprocess error details: {e}")
        raise

    # Step 2: Run metrics.py
    metrics_command = [
        "python", "metrics.py",
        "-m", model_folder_path
    ]
    try:
        subprocess.check_call(metrics_command, cwd=script_path)
        print(f"Metrics calculation completed for {model_folder_name}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Metrics calculation failed for {model_folder_name}")
        print(f"DEBUG: Subprocess error details: {e}")
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run render and metrics on model folders')
    parser.add_argument('--parent_folder', type=str, required=True, help='Path to the parent folder containing model folders')
    parser.add_argument('--colmap_dataset', type=str, required=True, help='Path to the colmap dataset')
    parser.add_argument('--script_path', type=str, required=True, help='Path to the script directory where render.py and metrics.py are located')
    args = parser.parse_args()

    # Step 1: Iterate through all folders in the parent folder
    for model_folder in os.listdir(args.parent_folder):
        model_folder_path = os.path.join(args.parent_folder, model_folder)
        
        # Ensure it's a directory before processing
        if os.path.isdir(model_folder_path):
            print(f"Processing folder: {model_folder_path}")
            try:
                # Run render and metrics for the current model folder
                run_render_and_metrics(args.script_path, model_folder_path, args.colmap_dataset)
            except Exception as e:
                print(f"An error occurred while processing {model_folder_path}: {e}")
