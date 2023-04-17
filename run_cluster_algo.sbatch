#!/bin/bash
#
#SBATCH --job-name=cluster_algorithm
#SBATCH --output=job_output/dendrogram.%j.out
#SBATCH --error=job_output/dendroErr.%j.err
#SBATCH --time=10:00
#SBATCH --nodes=1
##SBATCH --mem=50M

#TO_RUN="python3 geneExpressionClustering.py human_breast_cancer_data/*.txt Breast_Cancer_Data 3"
TO_RUN="python3 geneExpressionClustering.py mouse_data/*.txt Mouse_Data 2"
#TO_RUN="python3 geneExpressionClustering.py zebrafish_data/*.txt Zebrafish_Data 2"

echo
echo "BEGIN SCRIPT"
date
echo $TO_RUN
eval $TO_RUN
date
echo "SCRIPT COMPLETE"
echo
