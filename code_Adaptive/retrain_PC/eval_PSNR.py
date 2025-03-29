import os
import subprocess

# 设置根目录
base_dir = r'C:\Users\jay\Desktop\position_compressed'
dataset_dir = r'C:\Users\jay\Desktop\Dataset'
subdirs = ['train', 'truck', 'playroom']

def get_data_from_modelpath(modelpath):
    for data in subdirs:
        if data in modelpath.lower():
            return data
    return None

def run_command(command):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e)

def rename_ply_files_and_delete_cfg(model_path_full):
    for file in os.listdir(model_path_full):
        if file.endswith('.ply'):
            original_file_path = os.path.join(model_path_full, file)
            new_file_path = os.path.join(model_path_full, 'input.ply')
            os.rename(original_file_path, new_file_path)
            print(f"Renamed {original_file_path} to {new_file_path}")

    # 删除arg_cfg文件
    cfg_file_path = os.path.join(model_path_full, 'arg_cfg')
    if os.path.exists(cfg_file_path):
        os.remove(cfg_file_path)
        print(f"Deleted {cfg_file_path}")

def main():
    commands_to_run = []
    
    # 遍历所有文件夹并收集命令
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for modelpath in os.listdir(subdir_path):
                model_path_full = os.path.join(subdir_path, modelpath)
                if os.path.isdir(model_path_full):
                    data = get_data_from_modelpath(modelpath)
                    if data:
                        rename_ply_files_and_delete_cfg(model_path_full)
                        train_command = ['python', 'train.py', '-s', os.path.join(dataset_dir, data), '-m', model_path_full, '--eval']
                        render_command = ['python', 'render.py', '-s', os.path.join(dataset_dir, data), '-m', model_path_full]
                        metrics_command = ['python', 'metrics.py', '-m', model_path_full]
                        commands_to_run.append((train_command, render_command, metrics_command))
                        print(f"-s {os.path.join(dataset_dir, data)} -m {model_path_full}")


    
    # 运行收集的命令
    for train_command, render_command, metrics_command in commands_to_run:
        print(f"Running train command: {train_command}")
        run_command(train_command)
        print(f"Running render command: {render_command}")
        run_command(render_command)
        print(f"Running metrics command: {metrics_command}")
        run_command(metrics_command)

if __name__ == "__main__":
    main()
