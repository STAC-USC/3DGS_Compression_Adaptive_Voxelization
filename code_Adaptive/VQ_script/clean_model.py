from pathlib import Path
import os
import shutil
from argparse import ArgumentParser

def rename_and_cleanup(base_dir):
    subfolders = ["3DGS+VQ", "LightGaussian", "Original"]

    for subfolder in subfolders:
        subfolder_path = Path(base_dir) / subfolder

        if subfolder_path.is_dir():
            for model_dir in os.listdir(subfolder_path):
                model_path = subfolder_path / model_dir / "point_cloud" / "iteration_30000"
                if model_path.is_dir():
                    old_ply_file = model_path / "point_cloud.ply"
                    new_ply_file = model_path / "updated_point_cloud.ply"

                    if old_ply_file.is_file():
                        print(f"Deleting old PLY file: {old_ply_file}")
                        old_ply_file.unlink()  # 删除旧的 point_cloud.ply 文件

                    if new_ply_file.is_file():
                        print(f"Renaming {new_ply_file} to {old_ply_file}")
                        new_ply_file.rename(old_ply_file)  # 将 updated_point_cloud.ply 重命名为 point_cloud.ply

                    # 删除 test 文件夹
                    test_path = subfolder_path / model_dir / "test"
                    if test_path.is_dir():
                        print(f"Deleting test directory: {test_path}")
                        shutil.rmtree(test_path)

                    # 删除 train 文件夹
                    train_path = subfolder_path / model_dir / "train"
                    if train_path.is_dir():
                        print(f"Deleting train directory: {train_path}")
                        shutil.rmtree(train_path)

                    # 删除 results.json 文件
                    results_file = subfolder_path / model_dir / "results.json"
                    if results_file.is_file():
                        print(f"Deleting results.json file: {results_file}")
                        results_file.unlink()

if __name__ == "__main__":
    parser = ArgumentParser(description="Rename updated PLY files, delete old ones, and cleanup directories")
    parser.add_argument('--base_dir', required=True, type=str, help="Path to the base directory containing model directories")
    args = parser.parse_args()

    rename_and_cleanup(Path(args.base_dir))


