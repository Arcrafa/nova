#!/bin/sh
#SBATCH --qos=regular
#SBATCH --account=novapro
#SBATCH --job-name=CVN_BASE
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_gce
#SBATCH --nodelist=wcgpu06
#SBATCH --time=24:00:00

module load anaconda3
module load cuda11
conda activate /work1/nova/grohmc/conda/envs/nascvn/

cd /work1/nova/grohmc/CVN

python predict.py -c $1 --model_file $2
