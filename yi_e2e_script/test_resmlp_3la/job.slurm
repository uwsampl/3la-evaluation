#!/bin/bash
#SBATCH --job-name=3la_resmlp    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-124              # job array with index values 0, 1, 2, 3, 4
#SBATCH --partition=malik
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=yl29@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"

# module purge
# source $HOME/.ilang_env_rc
module purge

mkdir -p /scratch/test_resmlp_3la/batch_$SLURM_ARRAY_TASK_ID 
cd /scratch/test_resmlp_3la/batch_$SLURM_ARRAY_TASK_ID
echo "current dir is $PWD"

# use a separate python driver from dev
# export PY_3LA_DRIVER=/scratch/gpfs/yl29/3la_ILA_tensor_op
# export PYTHONPATH=/scratch/gpfs/yl29/3la_ILA_tensor_op:${PYTHONPATH}

python3 /home/yl29/slurm_job/test_resmlp_3la/test_resmlp.py
