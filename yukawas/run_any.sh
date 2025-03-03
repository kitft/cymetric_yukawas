#!/bin/bash
echo Running with arguments: "$@"
~/cymetric_yukawas/.venv/bin/python -u do_model.py "$@" "$SLURM_JOB_ID"
