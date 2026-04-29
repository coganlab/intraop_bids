#!/bin/bash
# bids_root="~/cworkspace/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS"
# task="phoneme"
# subj_arr=("S14" "S16" "S18" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")


bids_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
task="lexical"
subj_arr=("S52")
# subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81" "S83")
# subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81")
# subj_arr=("S41" "S45" "S47")
# subj_arr=("S56" "S63")


task_period="[perception,production]"
# task_period="perception"
# task_period="production"

# recompute="true"
recompute="false"

for subject in "${subj_arr[@]}"; do
    echo "----------------------------------------"
    echo "Creating processing job for ${subject}"
    job_name="${subject}_gen_specChanMap"
    sbatch -J "${job_name}" run_gen_specChanMap.sh -s "${subject}" -b "${bids_root}" -t "${task}" -p "${task_period}" -r "${recompute}"
    echo "----------------------------------------"
done