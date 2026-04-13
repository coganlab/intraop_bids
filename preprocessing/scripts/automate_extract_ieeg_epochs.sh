#!/bin/bash
# bids_root="~/cworkspace/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS"
# task="phoneme"
# subj_arr=("S14" "S16" "S18" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")


bids_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
task="lexical"
subj_arr=("S82")
# subj_arr=("S41" "S45" "S47" "S51" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81")
# subj_arr=("S41" "S45" "S47" "S51" "S53" "S55" "S56" "S63" "S67" "S74" "S75" "S76" "S78" "S81")
# subj_arr=("S56" "S63")


features="[high_gamma,spike_band]"
# features="[high_gamma]"
# features="[spike_band]"
task_period="[perception,production]"
# task_period="perception"
# task_period="production"
useSig=true  # leave this lower case
# useSig=false  # leave this lower case


for subject in "${subj_arr[@]}"; do
    echo "----------------------------------------"
    echo "Creating processing job for ${subject}"
    job_name="${subject}_extract_ieeg_epochs"
    sbatch -J "${job_name}" run_extract_ieeg_epochs.sh -s "${subject}" -b "${bids_root}" -t "${task}" -f "${features}" -p "${task_period}" -u "${useSig}"
    echo "----------------------------------------"
done