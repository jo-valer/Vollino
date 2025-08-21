#!/bin/sh
#SBATCH -p edu-medium
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
#SBATCH -N 1
#SBATCH -t 0-00:30
#SBATCH --output=runs/%x-%j.out

source /home/giovanni.valer/.bashrc

conda activate vollino

python3 pipeline.py llama3 --evaluate --avoid-input-translation --eval-nlu-only --user-id 1
