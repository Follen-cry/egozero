#!/bin/bash

if [ $LD_LIBRARY_PATH ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
fi

if [ $PATH ]; then
    export PATH=/usr/local/cuda/bin:$PATH
else
    export PATH=/usr/local/cuda/bin
fi

export CUDA_HOME=/usr/local/cuda-12.1

git submodule update --init --recursive

# deoxys
git clone https://github.com/NYU-robot-learning/deoxys_control.git src/deoxys_control
cd src/deoxys_control/deoxys
./InstallPackage  # enter 0.13.3 for the frankalib version when prompted
make -j build_deoxys=1
python -m pip install -e .
python -m pip install -U -r requirements.txt
cd ../../..

# franka
cd Franka-Teach
pip install -e .
pip install -r requirements.txt
cd ..

cd franka-env
python -m pip install -e .
cd ..

# requirements & aria
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip cache purge
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install projectaria_tools==1.6.0 --no-deps  # for some reason this can't be done at the same time as projectaria-client-sdk in requirements.txt
python -m pip cache purge
conda clean --all -y

# hamer
cd hamer/
bash fetch_demo_data.sh
rm hamer_demo_data.tar.gz
gdown "https://drive.google.com/uc?id=1x9dbp-H9u0V3j1ELu6jG6A_clLvpOfhx&confirm=t"
mv MANO_RIGHT.pkl _DATA/data/mano/
python -m pip install -e .[all]
python -m pip install -v -e third-party/ViTPose
cd ../

# devignetting masks
gdown "https://drive.google.com/uc?id=1V-MlvAgK0pBhRM53BziG4OuUFIRZkEIO&confirm=t"
tar -zxvf aria_devignetting_masks.tar.gz
rm aria_devignetting_masks.tar.gz

# cotracker
wget -P checkpoints https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
