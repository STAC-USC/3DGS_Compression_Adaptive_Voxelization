import os
import numpy as np
import json

def get_npz_file_paths(base_dir):
    npz_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'point_cloud.npz':
                npz_paths.append(os.path.join(root, file))
    return npz_paths

def calculate_bytestream(npz_path, model_path):
    state_dict = np.load(npz_path)
    bytestream_sizes = {}
    covariance_related = ['scaling', 'scaling_scale', 'scaling_zero_point', 
                          'scaling_factor', 'scaling_factor_scale', 'scaling_factor_zero_point', 
                          'rotation', 'rotation_scale', 'rotation_zero_point']
    covariance_size = 0
    total_size = 0

    for field in state_dict:
        value = state_dict[field]
        if isinstance(value, np.ndarray):
            size = value.nbytes
        elif isinstance(value, bool):
            size = 1  # Assuming 1 byte for a boolean value
        elif isinstance(value, int):
            size = 4  # Assuming 4 bytes for an integer value
        elif isinstance(value, float):
            size = 8  # Assuming 8 bytes for a float value
        else:
            size = 0  # Default size for unknown types
        
        bytestream_sizes[field] = size
        total_size += size
        if field in covariance_related:
            covariance_size += size
    
    print(f"Field sizes for {npz_path}:")
    for field, size in bytestream_sizes.items():
        print(f"{field}: {size} bytes")
    
    print(f"Total byte stream size: {total_size} bytes")
    print(f"Covariance-related byte stream size: {covariance_size} bytes")

    json_path = os.path.join(model_path, "bitstream_sizes.json")
    with open(json_path, 'w') as f:
        json.dump({
            "total_bitstream_size": total_size,
            "covariance_matrix_bitstream_size": covariance_size
        }, f, indent=4)

def main():
    base_dir = r'C:\Users\jay\Desktop\vq_3e-6'
    npz_paths = get_npz_file_paths(base_dir)

    for npz_path in npz_paths:
        model_path = os.path.dirname(npz_path)
        calculate_bytestream(npz_path, model_path)

if __name__ == "__main__":
    main()





