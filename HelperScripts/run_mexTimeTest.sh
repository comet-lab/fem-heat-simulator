#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=20g
#SBATCH -J "TimeTest"
#SBATCH -o /home/nepacheco/scratch/mexTimeTest-%j.log
#SBATCH -e /home/nepacheco/scratch/mexTimeTest-%j.log
#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:0

module load matlab/R2024a

matlab -nodisplay -nosplash -noFigureWindows -nodesktop -batch \
 "addpath('/home/nepacheco/Repositories/fem-heat-simulator/'); run('MexTesting/MultiStepTest.m');exit;"

