#!/bin/bash

bids_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
task="lexical"
# subj_arr=("S82" "S83")
# subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81" "S82" "S83")
subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81")

task_period="[perception,production]"
features="[high_gamma,spike_band]"

for subject in "${subj_arr[@]}"; do
    echo "----------------------------------------"
    echo "Creating feature video job for ${subject}"
    job_name="${subject}_gen_featureChanMapVids"
    sbatch -J "${job_name}" run_gen_featureChanMapVids.sh -s "${subject}" -b "${bids_root}" -t "${task}" -p "${task_period}" -f "${features}"
    echo "----------------------------------------"
done
