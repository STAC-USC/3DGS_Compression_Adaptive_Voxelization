import os
import subprocess

def run_commands():
    # Paths to the model and dataset directories
    model_base_path = r"C:\Users\jay\Desktop\3DGS_compression\test_model"
    dataset_base_path = r"C:\Users\jay\Desktop\Jay_pipeline\colmap_dataset"

    # Get all subfolders in the model directory
    model_folders = [f for f in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path, f))]

    # Iterate over each model folder
    for model_folder in model_folders:
        model_path = os.path.join(model_base_path, model_folder)
        dataset_name = model_folder.split("_")[0]  # Extract dataset name from the folder name
        dataset_path = os.path.join(dataset_base_path, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"[WARNING] Dataset path does not exist for {dataset_name}: {dataset_path}")
            continue

        # Define the commands
        commands = [
            [
                "python", "render.py",
                "-m", model_path,
                "-s", dataset_path,
                "--data_device", "cuda", "--eval"
            ],
            [
                "python", "metrics.py",
                "-m", model_path
            ]
        ]

        # Run each command
        for command in commands:
            print(f"Running command: {' '.join(command)}")
            try:
                subprocess.run(command, check=True)
                print("Command completed successfully.\n")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Command failed: {' '.join(command)}")
                print(f"Error: {e}\n")

if __name__ == "__main__":
    run_commands()
