#!/bin/sh
#SBATCH --job-name=rafael_test_k
### ignore #SBATCH --qos=regular
### ignore #SBATCH --account=nova
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_gce
### ignore #SBATCH --time=24:00:00
### ignore #SBATCH --constraint=v100
### ignore #SBATCH --nodelist=wcgpu06
### ignore #SBATCH --nodelist=wcgwn008

# Setup the singularity container

module load singularity
module load cuda10/10.1

export SINGULARITY_CACHEDIR=/scratch/.singularity/cache
export SINGULARITY_BIND="/wclustre"

cd /work1/nova/kmulder/AdCVN/

# Required Paths
script=scripts/cvn.py

config=configs/config.cfg

mode=train
#mode=evaluate
#mode=compare_pid_dist
#mode=delta_pid
#mode=ad_layer_ana

# Running this one first for gen of intermediate dataset

cmd="python3 ${script} ${mode} -c ${config}"

#echo "Running python3 ${script} ${mode} -c ${config} in"
pwd
#cmd="python3 ${script}"

singularity exec --userns --nv --workdir=/scratch/work/ --home=/work1/nova /work1/nova/singularity/scratch/singularity-ML-tf1.12-20191126/ $cmd

echo "Completed & exit"

exit
