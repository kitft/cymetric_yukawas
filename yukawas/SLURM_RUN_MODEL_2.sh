#!/bin/bash
# JOB HEADERS HERE

#SBATCH -c 48                # Number of cores (-c)
#SBATCH --gres=gpu:0       # Number of GPUs
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared # Partition to submit to
#SBATCH --mem=150000          # Memory pool for all cores in MB (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python
source activate cymetric

# run code
python Model_13_Do.py #1 #2
