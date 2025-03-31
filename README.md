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

### Requirement

#### - Conda environment 1 (gaussian_splatting)

#### - Conda environment 2 (NVS)

#### - Geometry-based Point Cloud Compression Codec

### Standard Folder Structure

### Instruction
