#!/bin/sh
#SBATCH --qos=opp
#SBATCH --account=nova
#SBATCH --job-name=rafa_cvn_conda_predict
#SBATCH --gres=gpu:2
#SBATCH --partition gpu_gce
#SBATCH --nodelist=wcgpu05
#SBATCH --time=05:00:00

module load condaforge/py39
module load cuda11
conda activate /work1/nova/grohmc/conda/envs/nascvn/

cd /work1/nova/rafaelma/myCVN

python contar.py -c default.conf
