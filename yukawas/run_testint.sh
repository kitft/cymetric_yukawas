#!/bin/bash
#SBATCH --output=test_integration_%1_%j_%2.out

echo Running with arguments: "$@"
~/cymetric_yukawas/.venv/bin/python -u test_integration.py "$@"
