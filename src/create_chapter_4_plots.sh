#!/bin/bash

""" Script to run all the numbers for the Experiemnts chapter of the
graph kernels survey. The data used is stored on
/cluster/work/borgw/graphkernels-review-results/ (the raw json), but
analyse_multiple will save the accuracy.csv and such files in
../results/ """

# TODO: make all of these save to an output folder that can be copied
# into the data folder of the latex file

# Step 1: Run analyse_multiple on all results to create the 
# accuracy and auroc files: 
# - accuracy.csv and accuracy_with_sdev.csv
# - auroc.csv and auroc.csv
echo "============================================================="
echo "Table 4.4"
poetry run python analyse_multiple.py /cluster/work/borgw/graphkernels-review-results/*.json -m 'accuracy'
echo "============================================================="

echo "============================================================="
echo "Table 4.5"
poetry run python analyse_multiple.py /cluster/work/borgw/graphkernels-review-results/*.json -m 'auroc'
echo "============================================================="

# Step 2: Update table 4.4 (accuracy)
# Copy the results that are printed in the terminal and paste into the
# latex doc
# TODO Separate EH, VH into separate table.

# Step 3: Update table 4.5 (auroc)
# Copy the results that are printed in the terminal and paste into the
# latex doc

# Step 4: Update Figure 4.3 (average depth of h in the WL framework).
# I don't need to upate this the moment, since none of the updated
# methods used WL, so I will keep it commented
# out.
# echo "Figure 4.3"
# poetry run python extract_average_depth.py

# Step 5: Update Table 4.6 (mean rank and sd for each kernel). This is
# based on simulating from their accuracy distributions. This must be
# typed in manually from the command output
echo "============================================================="
echo "Table 4.6"
poetry run python estimate_ranks.py ../results/accuracy_with_sdev.csv 
echo "============================================================="

# Step 6: Update Table 4.7 (ranking of kernels based on how often they
# outperform all other kernels) and Table 4.8 (ranking based on mean
# accuracy). This needs to be manually typed.
echo "============================================================="
echo "Table 4.7 and 4.8"
poetry run python count_winners_per_category.py ../results/accuracy.csv
echo "============================================================="

# Step 7: Update Figure 4.4 (accuracy values by dataset type). These
# output csvs need to be added to the document so tikz can render Figure
# 4.4. Creates output like "i_CSM.csv" in output/accuracies_per_class and needs to be copied into
#   Data/accuracies_per_class
echo "============================================================="
echo "Figure 4.4"
poetry run python partition_by_type.py ../results/accuracy.csv
echo "============================================================="

# Step 8: Update Figure 4.5 (mean accuracy of histogram kernel to best
# performing kernel). This is saved as a txt file
# (output/Vertex_histogram_kernel_{measure}_differences.txt) and put 
# here: Data/Vertex_histogram_kernel_accuracy_differences.txt and 
#  Data/Vertex_histogram_kernel_auroc_differences.txt. Had to manually
#  enter class category at the end of these txt.
echo "============================================================="
echo "Figure 4.5"
echo "accuracy"
poetry run python compare_baselines.py ../results/accuracy.csv
echo "auroc"
poetry run python compare_baselines.py ../results/auroc.csv
echo "============================================================="

# Step 9: Update Figure 4.6 (performance of VH against the best kernel,
# using accuracy) and Figure 4.7 (performance of VH against the best
# kernel in AUROC). This uses the output of Step 8, i.e. Figure 4.5,
# and the txt that is created using AUROC. Tikz generates the rest. 

# Step 10: Update Figure 4.8 (visualization of all accuracies including
# the s.d. on all benchmarkd datasets.)
echo "============================================================="
echo "Figure 4.6 and 4.7"
echo "I'm not sure if this is correct."
#poetry run python compare_baselines.py ../results/accuracy_with_sdev.csv
echo "============================================================="
# TODO: check this.

# Step 11: Update Figure 4.8 (all accuracies). This data is stored in 
# Data/accuracies/GH_accuracies.csv ????
echo "============================================================="
echo "Figure 4.8"
echo "I can't find this script. Need to write it"
#poetry run python collect_accuracies.py ../results/accuracy.csv
echo "============================================================="
# TODO: perhaps write this script? Can't find it

# Step 12: Update Figure 4.9 (accuracy across iterations, not
# highlighting EH/VH), which is stored here: Data/sdev/Synthie.txt. This
# uses the collect_accuracies_with_sdev and ends in txt. all the files
# need to be copied to the Data/sdev folder.
echo "============================================================="
echo "Figure 4.9"
poetry run python collect_accuracies_with_sdev.py ../results/accuracy_with_sdev.csv
echo "============================================================="

# Step 13: Update Figure 4.10 (a histogram of the fraction of pairwise
# overlaps). csv is in output/Overlaps.csv and must be saved into Data/Overlaps.csv
echo "============================================================="
echo "Figure 4.10"
poetry run python count_overlaps.py ../results/accuracy_with_sdev.csv
echo "============================================================="

# Step 14: Update Figure 4.11 (Boxplots of accuracy distribution). Data
# is stored in Data/Boxplots/name.txt
echo "============================================================="
echo "Figure 4.11"
poetry run python collect_accuracies.py ../results/accuracy.csv
echo "============================================================="

# Step 15: Update Table 4.9 (average performance of all GK on all
# datasets). The results get manually copied into the table.
echo "============================================================="
echo "Table 4.9"
poetry run python make_difficulty_table.py ../results/accuracy.csv
echo "============================================================="

# Step 16: Update Figure 4.12 (fraction of unclassifiable graphs). This
# is saved as Data/Difficulty_unclassifiable.csv
echo "============================================================="
echo "Figure 4.12 and Figure 4.13"
poetry run python collect_predictions.py 
poetry run python assess_difficulty.py  /cluster/work/borgw/graphkernels-review-results/*.json
echo "============================================================="
# TODO: Confirm this is working as planned. 

# Step 17: Update Figure 4.13 (difficult of dataset via best performance
# vs. unclassifiable). This comes from
# Data/Difficulty_unclassifiable.csv, and so I don't think any updates
# are required, since we can reuse the results from Step 16.

# Step 18: Update Figure 4.14 (critical difference plot). This is saved
# as a pdf in Overleaf, I'm not sure how this is generated.
echo "============================================================="
echo "Figure 4.14 Needs to be run in R."
echo "============================================================="
# TODO: Update R to be able to get this to work.
# TODO: Need to add in additional GK that were added.

# Step 19: Update Figure 4.15 (embedding of GKs according to distances).
# Uses a file Data/Kernel_predictions_MDS.txt and tikz renders the
# image.
echo "============================================================="
echo "Figure 4.15"
poetry run python embed_kernel_predictions.py ../results/Predictions.csv
echo "============================================================="
# TODO: What is the input file?? It is a dataframe, so maybe
# Predictions.csv?

# Step 20: Update Figure 4.16 (Dendrogram). Uses the output of
# embed_kernel_predictions.py (i.e. Distances_Hamming.csv). The two pdfs
# are generated in R and needs to be manually uploaded to overleaf
# (Dendrogram_AUROC.pdf, Dendrogram_Hamming.pdf).
echo "============================================================="
echo "Figure 4.16"
echo "Needs to be run in R"
echo "============================================================="
# TODO: Update it to include new GK!
# TODO: Do we still use the filtered version?i
# TODO: Update filtered_data_sets to save auroc_filtereed and not
# replace it.

# Step 21: Update Figure 4.18 (selection process)
echo "============================================================="
echo "Figure 4.18"
echo "Needs to be updated in Overleaf"
echo "============================================================="
# TODO: Update in Overleaf to include the new GK!
