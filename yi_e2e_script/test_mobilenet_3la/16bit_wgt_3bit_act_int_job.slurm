#!/bin/bash
#SBATCH --job-name=3la_mobilenet    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=23:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-124              # job array with index values 0, 1, 2, 3, 4
#SBATCH --partition=malik
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=yl29@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

# module purge
source $HOME/.ilang_env_rc

mkdir -p /scratch/test_mobilenet_3la_16bit/batch_$SLURM_ARRAY_TASK_ID 
cd /scratch/test_mobilenet_3la_16bit/batch_$SLURM_ARRAY_TASK_ID
echo "current dir is $PWD"

# use a separate python driver from dev
# export PY_3LA_DRIVER=/scratch/gpfs/yl29/3la_ILA_tensor_op
# export PYTHONPATH=/scratch/gpfs/yl29/3la_ILA_tensor_op:${PYTHONPATH}

export RESULT_OUTPUT_PATH=/home/yl29/gpfs/mobilenetv2_3la_16bit_wgt_3bit_int_act
# enable 16 bit weights
export HLSCNN_USE_16_WGT=1
export HLSCNN_FOR_MOBILENET=1
# disable debugging
unset HLSCNN_ILASIM_DEBUG
unset TVM_3LA_REF_RUN
unset TVM_3LA_DIFF_DEBUG

python3 /home/yl29/slurm_job/test_mobilenet_3la/test_mobilenet_slurm.py
