#!/bin/sh
#SBATCH --qos=opp
#SBATCH --account=nova
#SBATCH --job-name=build
#SBATCH --gres=gpu:2
#SBATCH --partition gpu_gce
#SBATCH --nodelist=wcgpu01,wcgpu02
#SBATCH --time=05:00:00

module load condaforge/py39
conda activate /work1/nova/grohmc/conda/envs/nascvn/

cd /work1/nova/rafaelma/cvn2

python build_test.py
