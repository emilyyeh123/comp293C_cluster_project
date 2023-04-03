#!/bin/bash
#
#SBATCH --job-name=cluster_algorithm
#SBATCH --output=job_output/dendrogram.%j.out
#SBATCH --error=job_output/dendroErr.%j.err
#SBATCH --time=10:00
#SBATCH --nodes=1
##SBATCH --mem=50M

TO_RUN="python3 hierarchical_cluster_algorithm.py"

echo
echo "BEGIN SCRIPT"
date
echo $TO_RUN
eval $TO_RUN
date
echo "SCRIPT COMPLETE"
echo
