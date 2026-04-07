#!/bin/bash

# Activate conda environment
conda activate ieeg

# Configuration
# source directory us the local box drive sync location of processed ecog data
# SOURCE_DIR="/Users/shinanlin/Library/CloudStorage/Box-Box/CoganLab/Data/Micro/Processed Data"
SOURCE_DIR="C:\Users\ns458\Box\CoganLab\Data\Micro\Processed Data"

# bids root is where you want the bids dataset to be created
BIDS_ROOT="./bids/lexical"
# task is the name of the task
TASK="lexical"
# audio directory is the directory containing the audio files
AUDIO_DIR="C:\Users\ns458\Box\CoganLab\Data\Micro\microphone"

ANAT_DIR="C:\Users\ns458\Box\ECoG_Recon"
# List of subjects to process
# SUBJECTS=(
#           "S14" "S16" "S18" "S22" 
#           "S23" "S26" "S32" "S33" 
#           "S36" "S39" "S57" "S58" 
#           "S62"
#           )

# SUBJECTS=(
#     "S41" "S45" "S47" "S51" "S52" 
#     "S53" "S55" "S56" "S63" "S67" 
#     "S71" "S73" "S74"
# )

SUBJECTS=(
    "S47"
)
# Process each subject
for SUBJECT in "${SUBJECTS[@]}"; do
    echo "Processing subject: $SUBJECT"

    python -m ecog_bids_pipeline.bids_writer \
        --source-dir "$SOURCE_DIR" \
        --bids-root "$BIDS_ROOT" \
        --subject "$SUBJECT" \
        --task "$TASK" \
        --audio-dir "$AUDIO_DIR" \
        --anat-dir "$ANAT_DIR"
    echo "Completed processing for subject: $SUBJECT"
done

echo "All BIDS conversions completed. Output in: $BIDS_ROOT"