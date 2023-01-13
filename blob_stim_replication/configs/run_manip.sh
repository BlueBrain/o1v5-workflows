#!/bin/bash -l
#SBATCH --job-name=conn_manip
#SBATCH --time=2:00:00
#SBATCH --account=proj9
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

module purge
module load archive/2021-07
source /gpfs/bbp.cscs.ch/home/pokorny/ToposampleKernel/bin/activate
python -u -m connectome_manipulator.connectome_manipulation $1 $2 $3 $4

# EXAMPLE HOW TO RUN: sbatch run_manip.sh <manip_config.json> [do_profiling] [do_resume] [keep_parquet]
# e.g. sbatch run_manip.sh manip_config.json 1 0 0