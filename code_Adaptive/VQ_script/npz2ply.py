from os import path
import numpy as np
import torch
from scene import GaussianModel
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement

def load_xyz_from_ply(ply_file):
    plydata = PlyData.read(ply_file)
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    return xyz

if __name__ == "__main__":
    parser = ArgumentParser("npz2ply")
    parser.add_argument("npz_file", type=str)
    parser.add_argument("--ply_file", type=str, default=None, required=False)
    parser.add_argument("--input_model_path", type=str, default=None, required=False)
    parser.add_argument("--iteration", type=int, default=35000, required=False)
    args = parser.parse_args()

    if args.ply_file is None:
        file_path = path.splitext(args.npz_file)[0]
        args.ply_file = f"{file_path}.ply"

    gaussians = GaussianModel(3)
    print(f"loading '{args.npz_file}'")
    gaussians.load_npz(args.npz_file)

    if args.input_model_path:
        input_ply_file = path.join(args.input_model_path, f"point_cloud/iteration_30000/point_cloud.ply")
        print(f"loading original point positions from '{input_ply_file}'")
        original_xyz = load_xyz_from_ply(input_ply_file)
        gaussians._xyz = torch.tensor(original_xyz, dtype=torch.float, device="cuda")

    print(f"saving to '{args.ply_file}'")
    gaussians.save_ply(args.ply_file)
    print("done")
