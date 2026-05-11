#!/bin/bash

bids_root="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
task="lexical"

subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81" "S82" "S83")
# subj_arr=("S82")

contact_sizes="[2,4,8]"


for subject in "${subj_arr[@]}"; do
    echo "----------------------------------------"
    echo "Creating spatial-avg job for ${subject} (contact_sizes=${contact_sizes})"
    job_name="${subject}_spatial_avg_raw"
    sbatch -J "${job_name}" run_spatial_avg_raw.sh \
        -s "${subject}" -b "${bids_root}" -t "${task}" \
        -z "${contact_sizes}"
    echo "----------------------------------------"
done
