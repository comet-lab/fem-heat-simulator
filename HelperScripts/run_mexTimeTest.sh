#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH -J "TimeTest"
#SBATCH -o /home/nepacheco/scratch/mexTimeTest-%j.log
#SBATCH -e /home/nepacheco/scratch/mexTimeTest-%j.log
#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1

# ------------------------------
# Paths to GCC 12 (Spack)
# ------------------------------
GCC12_PATH=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.4.0/gcc-12.1.0-i6yk33fh72z4rymfmydkenietcbpzz63
export LD_LIBRARY_PATH="$GCC12_PATH/lib64:$LD_LIBRARY_PATH"
export PATH="$GCC12_PATH/bin:$PATH"

# ------------------------------
# Load modules
# ------------------------------
module load matlab/R2024a
module load cuda/12.6.3

# ------------------------------
# Run MATLAB with LD_PRELOAD to force GCC 12 libstdc++
# ------------------------------
LD_PRELOAD=$GCC12_PATH/lib64/libstdc++.so.6 matlab -nodisplay -nosplash -noFigureWindows -nodesktop -batch \
 "addpath('/home/nepacheco/Repositories/fem-heat-simulator/'); run('MexTesting/MultiStepTest.m'); exit;"
