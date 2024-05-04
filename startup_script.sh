#!/bin/bash

# git clone patchfusion into models
# dpt.py needs to be replaced to avoid pathing
cd ../models
git clone https://github.com/zhyever/PatchFusion.git
# yml install from environment
cd PatchFusion
python ../../custom_node/pip_yaml.py
rm external/depth_anything/dpt.py
mv ../../custom_node/dpt.py external/depth_anything/dpt.py
pip install scikit-image
cd ../../custom_node

