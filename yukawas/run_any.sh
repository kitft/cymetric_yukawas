#!/bin/bash
#SBATCH --output=runmodel_%1_%2_%j_%3.out

echo Running with arguments: "$@"
~/cymetric_yukawas/.venv/bin/python -u do_model.py "$@" "$SLURM_JOB_ID"
