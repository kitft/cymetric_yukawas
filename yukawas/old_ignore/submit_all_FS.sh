#!/bin/bash

# Default values
memory=8
cores=10

# Function to display usage
usage() {
    echo "Usage: $0 [-m memory] [-c cores]"
    echo "  -m memory  : Memory in GB (default: 8)"
    echo "  -c cores   : Number of cores (default: 10)"
    exit 1
}

# Parse command-line options
while getopts ":m:c:" opt; do
    case $opt in
        m) memory=$OPTARG ;;
        c) cores=$OPTARG ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# Loop through numbers from 0 to 10 with 0.1 step
for number in $(seq 0.000000032000001 0.1 10.000000032000001); do
    # Format number to ensure two decimal places
    formatted_number=$(printf "%.2f" $number)

    # Submit the job
    addqueue -s -q long -n 1x$cores -m $memory ./run_FS_all.sh $formatted_number

    echo "Submitted job for number: $formatted_number"
done

echo "All jobs submitted."
