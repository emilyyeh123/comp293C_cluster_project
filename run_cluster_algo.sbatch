#!/bin/bash
#
#SBATCH --job-name=cluster_algorithm
#SBATCH --output=job_output/dendrogram.%j.out
#SBATCH --error=job_output/dendroErr.%j.err
#SBATCH --time=10:00
#SBATCH --nodes=1
##SBATCH --mem=50M

#TO_RUN="python3 hierarchical_cluster_algorithm.py"
#TO_RUN="python3 hierarchical_cluster_algorithm.py test_shrunken_dataset/*.txt dendrograms/testDendro.png"
#TO_RUN="python3 hierarchical_cluster_algorithm.py human_fibroblast_data/*.txt dendrograms/fibroblastDendrogram.png"
#TO_RUN="python3 hierarchical_cluster_algorithm.py human_lens_cataract_data/*.txt dendrograms/lensCataractDendrogram.png"
#TO_RUN="python3 hierarchical_cluster_algorithm.py human_breast_cancer_data/*.txt dendrograms/breastCancerDendrogram.png"
#TO_RUN="python3 hierarchical_cluster_algorithm.py human_blood_covid_data/*.txt dendrograms/bloodCovidDendrogram.png"

echo
echo "BEGIN SCRIPT"
date
echo $TO_RUN
eval $TO_RUN
date
echo "SCRIPT COMPLETE"
echo
