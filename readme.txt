# adaptive(uniform) voxelization + recoloring
conda activate gaussian_splatting
cd \repo\code_Adaptive
python voxelization.py --depth_start 15 --voxel_thr 30 --dataset_name train --retrain_mode 3DGS --use_adaptive false --iterations 15000
conda deactivate

# Lossy covariance compression pipeline
conda activate C:\Users\jay\AppData\Local\anaconda3\envs\NVS 
cd \repo\code_Adaptive\Lossy_covar
python codec.py --depth_start 15 --voxel_thr 30 --dataset_name train --retrain_mode 3DGS --use_adaptive false
conda deactivate 

# Lossless covariance compression pipeline
conda activate gaussian_splatting
cd \repo\code_Adaptive\Lossless_covar
python codec.py --depth_start 15 --voxel_thr 30 --dataset_name train --retrain_mode PC --use_adaptive false
conda deactivate 

# Rate Distortion analysis
conda activate gaussian_splatting
cd \repo\code_Adaptive|
python plot_RD.py
conda deactivate 