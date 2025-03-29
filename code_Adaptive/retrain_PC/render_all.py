import os
import subprocess
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置根目录
base_dir = r'C:\Users\jay\Desktop\Ablation_LT'
dataset_dir = r'C:\Users\jay\Desktop\Dataset'
subdirs = ['3DGS+VQ', 'LightGaussian', 'Original']

def get_data_from_modelpath(modelpath):
    for data in ['train', 'truck', 'playroom']:
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

def main():
    commands_to_run = []
    
    # 遍历所有文件夹并收集命令
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for modelpath in os.listdir(subdir_path):
                model_path_full = os.path.join(subdir_path, modelpath)
                if os.path.isdir(model_path_full):
                    data = get_data_from_modelpath(modelpath)
                    if data:
                        # 检查是否存在test文件夹
                        test_folder_path = os.path.join(model_path_full, 'test')
                        if not os.path.exists(test_folder_path):
                            # 构建命令
                            render_command = ['python', 'render.py', '-m', model_path_full, '-s', os.path.join(dataset_dir, data)]
                            metrics_command = ['python', 'metrics.py', '-m', model_path_full]
                            
                            commands_to_run.append((render_command, metrics_command))
                            
                            logging.info(f"Collected commands for model_path: {model_path_full}")
                            logging.info(f"Render command: {render_command}")
                            logging.info(f"Metrics command: {metrics_command}")
                        else:
                            logging.info(f"Skipping model_path: {model_path_full} because test folder exists.")

    # 打印所有将要运行的命令
    for render_command, metrics_command in commands_to_run:
        print(f"Render command: {render_command}")
        print(f"Metrics command: {metrics_command}")

    # 用户确认
    input("Press Enter to continue and run the commands...")

    # 运行收集的命令
    for render_command, metrics_command in commands_to_run:
        logging.info(f"Running render command: {render_command}")
        run_command(render_command)
        logging.info(f"Running metrics command: {metrics_command}")
        run_command(metrics_command)

if __name__ == "__main__":
    main()
