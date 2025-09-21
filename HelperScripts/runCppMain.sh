#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH --mem=20g
#SBATCH -J "FEM"
#SBATCH -o /home/nepacheco/scratch/job-%j.log
#SBATCH -e /home/nepacheco/scratch/job-%j.log
#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=10
module load cuda/12.8.0

./build/main
