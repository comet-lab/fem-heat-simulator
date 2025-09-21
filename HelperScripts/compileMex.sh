#!/bin/bash
module load gcc/12.1.0
module load cmake
module load eigen
module load matlab
module load cuda/12.6.3


cd ~/Repositories/fem-heat-simulator
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DAMGX_ROOT=/home/nepacheco/amgx/ ..
cmake --build .
cmake --install .

module unload gcc/12.1.0
module unload cmake
module unload eigen
module unload matlab
module unload cuda/12.6.3
