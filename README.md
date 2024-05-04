# Comfyui patchfusion depth
[WIP] Quick comfyui wrapper for patchfusion depth.

## Installation 
This should be simplified shortly
1. git clone this repo into the `custom_nodes` of comfyui
2. move all files from comfyui into `custom_nodes` (installation script expects to be run from here)
3. patchfusion.py must be in `custom_nodes` directory
4. run installion `startup_script.sh` (or you can run each line)
5. remove `dpt.py`, `pip_yaml.py`, `startup_script.sh` from `custom_nodes` (might cause conflicts)

## Usage
1. restarting comfyui should add a patchfusion node
2. you may want to add resize nodes before or after.

## Notes
-This repo does save some images into a temp folder

## Acknowledgement
This uses this paper/code
```
@article{li2023patchfusion,
    title={PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation}, 
    author={Zhenyu Li and Shariq Farooq Bhat and Peter Wonka},
    booktitle={CVPR},
    year={2024}
}
```
