#!/bin/bash -l
#SBATCH --job-name=conn_rewire
#SBATCH --time=1:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

module purge
module load unstable
module load parquet-converters/0.8.0
source /gpfs/bbp.cscs.ch/home/pokorny/ReWiringKernel/bin/activate

connectome-manipulator -v manipulate-connectome $1 --output-dir=$2 --profile --convert-to-sonata --splits=$3

# EXAMPLE HOW TO RUN: sbatch run_rewiring.sh <model_config.json> <output_dir> <num_splits>
# e.g. sbatch run_rewiring.sh manip_config.json /gpfs/.../O1v5-SONATA__Rewired 100
