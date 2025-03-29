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

def evaluate(playroom_dir, truck_dir, train_dir, subdir_path):
    results = {}
    print(f"Starting evaluation with playroom_dir={playroom_dir}, truck_dir={truck_dir}, train_dir={train_dir}, subdir_path={subdir_path}")

    render_dirs = {
        'playroom': playroom_dir,
        'truck': truck_dir,
        'train': train_dir
    }

    for model_dir in os.listdir(subdir_path):
        model_path = Path(subdir_path) / model_dir
        render_dir2 = model_path / "test/ours_35000/renders"
        print(f"Checking model_dir={model_dir}, render_dir2={render_dir2}")

        source_type = None
        if 'playroom' in model_dir.lower():
            source_type = 'playroom'
        elif 'truck' in model_dir.lower():
            source_type = 'truck'
        elif 'train' in model_dir.lower():
            source_type = 'train'

        if source_type:
            render_dir1 = render_dirs[source_type]
            if render_dir2.is_dir():
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
                print(f"Directory does not exist: {render_dir2}")
        else:
            print(f"Skipping {model_dir} as it does not match any source type")

    with open("difference.json", 'w') as fp:
        json.dump(results, fp, indent=4)
    print("Evaluation completed and results saved to difference.json")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Metrics evaluation parameters")
    parser.add_argument('--playroom_dir', required=True, type=str, help="Path to the playroom render directory")
    parser.add_argument('--truck_dir', required=True, type=str, help="Path to the truck render directory")
    parser.add_argument('--train_dir', required=True, type=str, help="Path to the train render directory")
    parser.add_argument('--subdir_path', required=True, type=str, help="Path to the directory containing multiple model directories")
    args = parser.parse_args()

    evaluate(Path(args.playroom_dir), Path(args.truck_dir), Path(args.train_dir), Path(args.subdir_path))

