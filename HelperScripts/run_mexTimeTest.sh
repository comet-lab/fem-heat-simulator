#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20g
#SBATCH -J "TimeTest"
#SBATCH -o /home/nepacheco/scratch/mexTimeTest-%j.log
#SBATCH -e /home/nepacheco/scratch/mexTimeTest-%j.log
#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1

$use_gpu=false
$use_all_cpus=false
$alpha=0.5

# ------------------------------
# Paths to GCC 12 (Spack)
# ------------------------------
GCC12_LIB=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.4.0/gcc-12.1.0-i6yk33fh72z4rymfmydkenietcbpzz63/lib64/libstdc++.so.6

# ------------------------------
# Load modules
# ------------------------------
module load matlab/R2024a
module load cuda/12.6.3

# ------------------------------
# Run MATLAB with LD_PRELOAD to force GCC 12 libstdc++
# ------------------------------
LD_PRELOAD=$GCC12_LIB matlab -nodisplay -nosplash -noFigureWindows -nodesktop -batch \
 "addpath(genpath('/home/nepacheco/Repositories/fem-heat-simulator/')); runMexTimeTest($use_all_cpus,$use_gpu,$alpha); exit;"

