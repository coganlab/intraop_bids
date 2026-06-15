#!/bin/bash

bids_root="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
task="lexical"

subj_arr=("S83")
# subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81" "S82" "S83")
# subj_arr=("S82")

contact_sizes="[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]"
# contact_sizes="[2]"

# Set to "true" to drop fully-NaN rows/columns of the chanmap before
# averaging (needed for multi-array recordings like S83 where a NaN
# gap row would otherwise prevent larger contacts from tiling cleanly
# across the gap).
compact_chanmap="true"


for subject in "${subj_arr[@]}"; do
    echo "----------------------------------------"
    echo "Creating spatial-avg job for ${subject} (contact_sizes=${contact_sizes}, compact_chanmap=${compact_chanmap})"
    job_name="${subject}_spatial_avg_raw"
    sbatch -J "${job_name}" run_spatial_avg_raw.sh \
        -s "${subject}" -b "${bids_root}" -t "${task}" \
        -z "${contact_sizes}" -c "${compact_chanmap}"
    echo "----------------------------------------"
done
