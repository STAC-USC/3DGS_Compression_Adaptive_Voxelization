import os
import subprocess
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置根目录
base_dir = r'C:\Users\jay\Desktop\R-PSNR_experiment\Ablation_LT'
dataset_dir = r'C:\Users\jay\Desktop\Dataset'
output_vq_dir = r'C:\Users\jay\Desktop\output_vq'
subdirs = ['train', 'truck', 'playroom']

gaussian_importance_values = [3e-7, 1.5e-6, 6e-6, 3e-5]

def get_data_from_modelpath(modelpath):
    for data in subdirs:
        if data in modelpath.lower():
            return data
    return None

def run_command(command):
    try:
        subprocess.run(command, check=True)
        logging.info(f"Successfully ran command: {command}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command: {command}")
        logging.error(e)

def rename_ply_files_and_delete_cfg(model_path_full):
    for file in os.listdir(model_path_full):
        if file.endswith('.ply'):
            original_file_path = os.path.join(model_path_full, file)
            new_file_path = os.path.join(model_path_full, 'input.ply')
            os.rename(original_file_path, new_file_path)
            logging.info(f"Renamed {original_file_path} to {new_file_path}")

    # 删除arg_cfg文件
    cfg_file_path = os.path.join(model_path_full, 'arg_cfg')
    if os.path.exists(cfg_file_path):
        os.remove(cfg_file_path)
        logging.info(f"Deleted {cfg_file_path}")

def create_output_vq_dirs(model_path_full, importance_value):
    relative_path = os.path.relpath(model_path_full, base_dir)
    parent_dir, model_dir = os.path.split(relative_path)
    new_relative_path = os.path.join(parent_dir, str(importance_value), model_dir)
    output_vq_path = os.path.join(output_vq_dir, new_relative_path)
    os.makedirs(output_vq_path, exist_ok=True)
    logging.info(f"Created output_vq directory: {output_vq_path}")
    return output_vq_path

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
                        
                        for importance_value in gaussian_importance_values:
                            # 创建 output_vq 目录结构
                            output_vq_path = create_output_vq_dirs(model_path_full, importance_value)
                            
                            # 构建命令
                            compress_command = [
                                'python', 'compress.py', '--model_path', model_path_full, '--data_device', 'cuda',
                                '--source_path', os.path.join(dataset_dir, data), '--output_vq', output_vq_path,
                                '--gaussian_importance_include', str(importance_value)
                            ]
                            render_command = ['python', 'render.py', '-m', output_vq_path, '-s', os.path.join(dataset_dir, data)]
                            metrics_command = ['python', 'metrics.py', '-m', output_vq_path]
                            npz_file_path = os.path.join(output_vq_path, 'point_cloud', 'iteration_35000', 'point_cloud.npz')
                            ply_file_path = os.path.splitext(npz_file_path)[0] + '.ply'
                            npz2ply_command = ['python', 'npz2ply.py', npz_file_path, '--ply_file', ply_file_path]
                            
                            commands_to_run.append((compress_command, render_command, metrics_command, npz2ply_command))
                            
                            logging.info(f"Collected commands for model_path: {model_path_full}")
                            logging.info(f"Compress command: {compress_command}")
                            logging.info(f"Render command: {render_command}")
                            logging.info(f"Metrics command: {metrics_command}")
                            logging.info(f"Npz2ply command: {npz2ply_command}")

    # 打印所有将要运行的命令
    for compress_command, render_command, metrics_command, npz2ply_command in commands_to_run:
        print(f"Compress command: {compress_command}")
        print(f"Render command: {render_command}")
        print(f"Metrics command: {metrics_command}")
        print(f"Npz2ply command: {npz2ply_command}")

    # 用户确认
    input("Press Enter to continue and run the commands...")

    # 运行收集的命令
    for compress_command, render_command, metrics_command, npz2ply_command in commands_to_run:
        logging.info(f"Running compress command: {compress_command}")
        run_command(compress_command)
        logging.info(f"Running render command: {render_command}")
        run_command(render_command)
        logging.info(f"Running metrics command: {metrics_command}")
        run_command(metrics_command)
        logging.info(f"Running npz2ply command: {npz2ply_command}")
        run_command(npz2ply_command)

if __name__ == "__main__":
    main()
