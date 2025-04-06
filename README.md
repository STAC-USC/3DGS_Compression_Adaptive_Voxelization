# A GPCC-based Compression Framework for 3D Gaussian Splatting

**Chenjunjie Wang\*, Shashank N. Sridhara\*, Eduardo Pavez\*, Antonio Ortega\*, Cheng Chang†**  
\*University of Southern California, Los Angeles, CA, 
†Meta, Menlo Park, CA  

![3DGS Demo](images/pipeline.png)

## Abstract

We present a novel compression framework for 3D Gaussian splatting (3DGS) data that leverages transform coding tools originally developed for point clouds. Contrary to existing 3DGS compression methods, our approach can produce compressed 3DGS models at multiple bitrates in a computationally efficient way. Point cloud voxelization is a discretization technique that point cloud codecs use to improve coding efficiency while enabling the use of fast transform coding algorithms. We propose an adaptive voxelization algorithm tailored to 3DGS data, to avoid the inefficiencies introduced by uniform voxelization used in point cloud codecs. We ensure the positions of larger volume Gaussians are represented at high resolution, as these significantly impact rendering quality. Meanwhile, a low-resolution representation is used for dense regions with smaller Gaussians, which have a relatively lower impact on rendering quality. This adaptive voxelization approach significantly reduces the number of Gaussians and the bitrate required to encode the 3DGS data. After voxelization, many Gaussians are moved or eliminated. Thus, we propose to fine-tune/recolor the remaining 3DGS attributes with an initialization that can reduce the amount of retraining required. Experimental results on pre-trained datasets show that our proposed compression framework outperforms existing methods.

---

## Keywords

- 3D Gaussian Splatting  
- Adaptive Voxelization  
- 3D Data Compression  
- Point Cloud Compression

---

## Installation

### 🛠️ Requirements

- **Conda Environment 1 (gaussian_splatting)**  
  👉 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

- **Conda Environment 2 (NVS)**  
  👉 [KeKsBoTer/c3dgs](https://github.com/KeKsBoTer/c3dgs/tree/master)

- **Geometry-based Point Cloud Compression (GPCC) Codec**  
  👉 [MPEGGroup/mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13)


### Standard Folder Structure
```text
📁 project_root/

├── 📁 attributes_compressed/
├── 📁 code_Adaptive/
│   ├── 📁 Lossless_covar/
│   │   ├── extract_all_pq.py
│   │   ├── postprocess.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── codec.py
│   ├── 📁 Lossless_covar/
│   │   ├── extract_all_pq.py
│   │   ├── postprocess.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── codec.py
│   ├── 📁 Retrain_3DGS/
│   │   ├── train.py
│   │   ├── render.py
│   │   ├── metrics.py
│   │   └── adapt_voxel_recolor.py
│   ├── 📁 Retrain_PC/
│   │   ├── train.py
│   │   ├── render.py
│   │   └── metrics.py
│   ├── 📁 VQ_script/
│   │   ├── compress.py
│   │   ├── render.py
│   │   └── metrics.py
│   ├── plot_RD.py
│   └── voxelization.py
├── 📁 colmap_dataset/
│   ├── 📁 bicycle/
│   ├── 📁 bonsai/
│   ├── 📁 counter/
│   ├── 📁 drjohnson/
│   ├── 📁 flowers/
│   ├── 📁 garden/
│   ├── 📁 kitchen/
│   ├── 📁 playroom/
│   ├── 📁 room/
│   ├── 📁 stump/
│   ├── 📁 train/
│   ├── 📁 treehill/
│   └── 📁 truck/
├── 📁 original_model/
│   ├── 📁 bicycle/
│   ├── 📁 bonsai/
│   ├── 📁 counter/
│   ├── 📁 drjohnson/
│   ├── 📁 flowers/
│   ├── 📁 garden/
│   ├── 📁 kitchen/
│   ├── 📁 playroom/
│   ├── 📁 room/
│   ├── 📁 stump/
│   ├── 📁 train/
│   ├── 📁 treehill/
│   └── 📁 truck/
├── 📁 test_model/
│   ├── 📁 bicycle/
│   ├── 📁 bonsai/
│   ├── 📁 counter/
│   ├── 📁 drjohnson/
│   ├── 📁 flowers/
│   ├── 📁 garden/
│   ├── 📁 kitchen/
│   ├── 📁 playroom/
│   ├── 📁 room/
│   ├── 📁 stump/
│   ├── 📁 train/
│   ├── 📁 treehill/
│   └── 📁 truck/
├── 📁 RDO/
│   ├── 📁 bpp/
│   ├── 📁 Meta_data/
│   ├── 📁 PSNR/
│   └── 📁 PSNR_per_view/
├── 📁 reconstructed_3DGS/
├── 📁 retrain_model/
├── 📁 voxelized_adapt/
├── 📁 VQ_model/
└── README.md
```


## 📘 Instruction

```bash
# =========================
# Step 1: Configure Conda Environments
# =========================
# Setup two Conda environments:
# - gaussian_splatting: for retraining the 3DGS model.
# - NVS: for VQ-based covariance compression.

# =========================
# Step 2: Install GPCC Codec
# =========================
# Clone and compile GPCC (TMC13) from:
# https://github.com/MPEGGroup/mpeg-pcc-tmc13
# Note: We build it with Visual Studio on Windows 10.
# After building, the executable will be located at:
#   /build/tmc3/Release/tmc3.exe

# =========================
# Step 3: Voxelization
# =========================
conda activate gaussian_splatting
cd code_Adaptive
python voxelization.py --depth_start 15 --voxel_thr 30 --dataset_name train --retrain_mode 3DGS --use_adaptive false --iterations 15000
conda deactivate

# =========================
# Step 4a: Lossy Compression
# =========================
# Quantization parameters are configured inside:
#   code_Adaptive/Lossy_covar/encoder.py
#   code_Adaptive/Lossy_covar/decoder.py
# Default QP combination: (f_rest_qp, f_dc_qp, opacity_qp) = (4, 4, 4)

conda activate NVS
cd code_Adaptive/Lossy_covar
python codec.py --depth_start 15 --voxel_thr 30 --dataset_name train --retrain_mode 3DGS --use_adaptive false
conda deactivate

# =========================
# Step 4b: Lossless Compression
# =========================
# Quantization parameters are configured inside:
#   code_Adaptive/Lossless_covar/encoder.py
#   code_Adaptive/Lossless_covar/decoder.py
# Default QP combination: (f_rest_qp, f_dc_qp, opacity_qp) = (4, 4, 4)

conda activate gaussian_splatting
cd code_Adaptive/Lossless_covar
python codec.py --depth_start 15 --voxel_thr 30 --dataset_name train --retrain_mode 3DGS --use_adaptive false
conda deactivate

# =========================
# Step 5: Rendering & RD Analysis
# =========================
# Evaluate the rate-distortion performance of compressed 3DGS.

conda activate gaussian_splatting
cd code_Adaptive
python plot_RD.py --depth_start 15 --voxel_thr 30 --dataset_name train --retrain_mode 3DGS --use_adaptive false --comp_mode lossless
conda deactivate
