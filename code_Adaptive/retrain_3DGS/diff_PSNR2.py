# metrics.py

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(render_dir1, render_dir2):
    renders1 = []
    renders2 = []
    image_names = []
    for fname in os.listdir(render_dir1):
        render1 = Image.open(render_dir1 / fname)
        render2 = Image.open(render_dir2 / fname)
        renders1.append(tf.to_tensor(render1).unsqueeze(0)[:, :3, :, :].cuda())
        renders2.append(tf.to_tensor(render2).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders1, renders2, image_names

def save_diff_image(image1, image2, output_path):
    diff = torch.abs(image1 - image2).squeeze(0)
    diff_image = transforms.ToPILImage()(diff.cpu())
    diff_image.save(output_path)

def delete_diff_images(directory):
    for file in os.listdir(directory):
        if file.startswith("diff_"):
            file_path = Path(directory) / file
            os.remove(file_path)
            print(f"Deleted {file_path}")

def evaluate(base_dir1, base_dir2):
    results = {}
    print(f"Starting evaluation with base_dir1={base_dir1}, base_dir2={base_dir2}")

    for model_dir in os.listdir(base_dir1):
        render_dir1 = Path(base_dir1) / model_dir / "test" / "ours_35000" / "renders"
        render_dir2 = Path(base_dir2) / model_dir / "test" / "ours_30000" / "renders"

        print(f"Checking render_dir1={render_dir1}, render_dir2={render_dir2}")

        if render_dir1.is_dir() and render_dir2.is_dir():
            # 删除 render_dir1 和 render_dir2 中的 diff_ 开头的图片
            delete_diff_images(render_dir1)
            delete_diff_images(render_dir2)

            print(f"Evaluating renders between {render_dir1} and {render_dir2}")

            try:
                renders1, renders2, image_names = readImages(render_dir1, render_dir2)
                print(f"Read {len(renders1)} images from {render_dir1} and {len(renders2)} images from {render_dir2}")

                psnrs = []

                for idx in tqdm(range(len(renders1)), desc="Metric evaluation progress"):
                    psnrs.append(psnr(renders1[idx], renders2[idx]))

                    # Save diff image
                    diff_output_path = render_dir2 / f"diff_{image_names[idx]}"
                    save_diff_image(renders1[idx], renders2[idx], diff_output_path)

                avg_psnr = torch.tensor(psnrs).mean().item()
                results[str(render_dir2)] = {"PSNR": avg_psnr}

                print(f"  Average PSNR for {render_dir2}: {avg_psnr:.7f}")
            except Exception as e:
                print(f"Unable to compute metrics for {render_dir2}: {e}")
        else:
            print(f"One or both directories do not exist: {render_dir1}, {render_dir2}")

    with open("difference.json", 'w') as fp:
        json.dump(results, fp, indent=4)
    print("Evaluation completed and results saved to difference.json")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Metrics evaluation parameters")
    parser.add_argument('--base_dir1', required=True, type=str, help="Path to the first base directory containing model directories")
    parser.add_argument('--base_dir2', required=True, type=str, help="Path to the second base directory containing model directories")
    args = parser.parse_args()

    evaluate(Path(args.base_dir1), Path(args.base_dir2))


