import os
import sys
import json
import re
import argparse
import subprocess

# Base path input
current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = base_path = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
base_path = os.path.abspath(os.path.join(root_path, 'RDO'))
bpp_path = os.path.join(base_path, 'bpp')

# Regex to match JSON file names
FILENAME_PATTERN = re.compile(
    r'^(?P<dataset_name>.+)_depth_(?P<depth>\d+)_rest_pq_(?P<rest_pq>\d+)_dc_pq_(?P<dc_pq>\d+)_opacity_pq_(?P<opacity_pq>\d+)\.json$'
)

# Regex to match REST file names
REST_FILENAME_PATTERN = re.compile(
    r'^(?P<dataset_name>.+)_depth_(?P<depth>\d+)_rest_(\d+_\d+_\d+)_pq_(?P<rest_pq>\d+)\.txt$'
)

def extract_bitstream_data(dataset_name, depth, rest_pq, dc_pq, opacity_pq, thr, mode, suffix, comp_mode, meta_size):
    """Extract bitstream data from bpp folder and process it, adding the meta data file size to the total bitstream size."""
    key = f"{dataset_name}_depth_{depth}_rest_pq_{rest_pq}_dc_pq_{dc_pq}_opacity_pq_{opacity_pq}"
    result = {'positions_B': 0, 'covariance_B': 0, 'dc_B': 0, 'opacity_B': 0, 'rest_B': 0}

    # Extract scale and rot files and accumulate them
    for i in range(3):  # Iterate through scale_0, scale_1, scale_2
        scale_file = os.path.join(bpp_path, f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless", f"{dataset_name}_depth_{depth}_scale_{i}_comp.txt")
        if os.path.exists(scale_file):
            with open(scale_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match_reflectance = re.match(r'reflectances bitstream size (\d+) B', line.strip())
                    if match_reflectance:
                        result['covariance_B'] += int(match_reflectance.group(1))
        else:
            print(f"Scale file not found: {scale_file}")

    for i in range(4):  # Iterate through rot_0, rot_1, rot_2, rot_3
        rot_file = os.path.join(bpp_path, f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless", f"{dataset_name}_depth_{depth}_rot_{i}_comp.txt")
        if os.path.exists(rot_file):
            with open(rot_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match_reflectance = re.match(r'reflectances bitstream size (\d+) B', line.strip())
                    if match_reflectance:
                        result['covariance_B'] += int(match_reflectance.group(1))
        else:
            print(f"Rot file not found: {rot_file}")

    # Process dc.txt file
    dc_file = os.path.join(bpp_path, f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless", f"{dataset_name}_depth_{depth}_dc_pq_{dc_pq}.txt")
    if os.path.exists(dc_file):
        with open(dc_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                match_pos = re.match(r'positions bitstream size (\d+) B', line)
                match_colors = re.match(r'colors bitstream size (\d+) B', line)
                if match_pos:
                    result['positions_B'] += int(match_pos.group(1))
                if match_colors:
                    result['dc_B'] += int(match_colors.group(1))
    else:
        print(f"DC file not found: {dc_file}")

    # Process opacity.txt file
    opacity_file = os.path.join(bpp_path, f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless", f"{dataset_name}_depth_{depth}_opacity_pq_{opacity_pq}.txt")
    if os.path.exists(opacity_file):
        with open(opacity_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r'reflectances bitstream size (\d+) B', line.strip())
                if match:
                    result['opacity_B'] += int(match.group(1))
    else:
        print(f"Opacity file not found: {opacity_file}")

    # Process multiple REST files and accumulate bitstream sizes (ensure matching dataset_name and depth)
    rest_path = os.path.join(bpp_path, f"{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless")
    rest_files = [
        f for f in os.listdir(rest_path)
        if REST_FILENAME_PATTERN.match(f) and
           int(REST_FILENAME_PATTERN.match(f).group('rest_pq')) == rest_pq and
           REST_FILENAME_PATTERN.match(f).group('dataset_name') == dataset_name and
           int(REST_FILENAME_PATTERN.match(f).group('depth')) == depth
    ]

    for rest_file in rest_files:
        with open(os.path.join(rest_path, rest_file), 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r'colors bitstream size (\d+) B', line.strip())
                if match:
                    result['rest_B'] += int(match.group(1))

    # Calculate total bitstream size and convert to MB and Mbits
    total_bytes = (
        result['positions_B'] + result['covariance_B'] +
        result['dc_B'] + result['opacity_B'] + result['rest_B'] + meta_size
    )
    result['meta_data_size_B'] = meta_size 
    result['total_bitstream_size_MB'] = total_bytes / (1024 * 1024)
    result['total_bitstream_size_Mbits'] = total_bytes * 8 / (1024 * 1024)

    return result


def merge_psnr_files(psnr_directory, output_filename, thr, mode, suffix, comp_mode, meta_size):
    """Merge JSON files in the PSNR folder and combine them with the extracted bitstream data."""
    merged_data = {}

    for filename in os.listdir(psnr_directory):
        if filename.endswith('.json'):
            print(f"Processing file: {filename}")
            match = FILENAME_PATTERN.match(filename)
            if match:
                # Extract information from the filename
                info = match.groupdict()
                dataset_name = info['dataset_name']
                depth = int(info['depth'])
                rest_pq = int(info['rest_pq'])
                dc_pq = int(info['dc_pq'])
                opacity_pq = int(info['opacity_pq'])
                print(f"Extracted info: {info}")
                file_path = os.path.join(psnr_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    psnr_data = json.load(f)
                bitstream_data = extract_bitstream_data(
                    dataset_name, depth, rest_pq, dc_pq, opacity_pq,
                    thr, mode, suffix, comp_mode, meta_size
                )
                merged_key = f"{dataset_name}_depth_{depth}_rest_pq_{rest_pq}_dc_pq_{dc_pq}_opacity_pq_{opacity_pq}"
                merged_data[merged_key] = {
                    'psnr': psnr_data,
                    'bitstream': bitstream_data
                }
            else:
                print(f"Filename did not match pattern: {filename}. Expected pattern: {FILENAME_PATTERN.pattern}")
    output_file_path = os.path.join(base_path, output_filename)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4)
    print(f"Merged PSNR files saved to {output_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Decompress, reorder, and process PLY files with given pq combinations.'
    )
    parser.add_argument('--depth_start', type=int, required=True, help='Initial depth for voxelization')
    parser.add_argument('--voxel_thr', type=int, required=True, help='Threshold for small voxel point count')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--retrain_mode', type=str, required=True,
                        help='Set to "PC" for retrain PC, "3DGS" for retrain 3DGS')
    parser.add_argument('--use_adaptive', type=str, required=True,
                        help='Set to "true" for adaptive voxelization, "false" for uniform voxelization')
    parser.add_argument('--comp_mode', type=str, required=True,
                        help='Set to "lossy" for lossy covariance compression, "lossless" for lossless covariance compression')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    depth = args.depth_start
    thr = args.voxel_thr
    # Convert retrain_mode to uppercase, e.g., "PC" or "3DGS"
    mode = args.retrain_mode.upper()
    suffix = "adapt" if args.use_adaptive.lower() == "true" else "uniform"
    
    # Read the size of the meta data file, assumed to be stored in base_path/Meta_data
    meta_data_file = os.path.abspath(os.path.join(base_path, "Meta_data", f"meta_data_{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossy.json"))
    if os.path.exists(meta_data_file):
        meta_size = os.path.getsize(meta_data_file)
        print(f"[DEBUG] Meta data file size: {meta_size} bytes")
    else:
        print(f"[WARNING] Meta data file not found: {meta_data_file}")
        meta_size = 0

    # Merge JSON files in the PSNR folder (assumed to be in base_path/PSNR/{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless)
    psnr_directory = os.path.join(base_path, 'PSNR', f'{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless')
    merge_psnr_files(
        psnr_directory,
        f'PSNR_{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless.json',
        thr, mode, suffix, meta_size
    )

    # Merge JSON files in the PSNR_per_view folder (assumed to be in base_path/PSNR_per_view/{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless)
    psnr_per_view_directory = os.path.join(base_path, 'PSNR_per_view', f'{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless')
    merge_psnr_files(
        psnr_per_view_directory,
        f'PSNR_per_view_{dataset_name}_depth_{depth}_thr_{thr}_{mode}_{suffix}_lossless.json',
        thr, mode, suffix, meta_size
    )

if __name__ == "__main__":
    main()
