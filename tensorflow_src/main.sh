#!/bin/sh
#SBATCH -J main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o main.out
#SBATCH -e main.err
#SBATCH --time 00:30:00
#SBATCH --gres=gpu:1

python3 main.py -save=temp