#!/bin/sh
#SBATCH -J main
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o main.out
#SBATCH -e main.err
#SBATCH --time 0
#SBATCH --gres=gpu:1

make eval include_more=1