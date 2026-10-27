#!/bin/bash
# bids_root="~/cworkspace/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS"
# task="phoneme"
# subj_arr=("S14" "S16" "S18" "S19" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")
# # subj_arr=("S14")


# bids_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
# task="lexical"
# subj_arr=("S84a" "S84b")
# # subj_arr=("S30" "S82" "S83")
# # subj_arr=("S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81")
# # subj_arr=("S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81")
# # subj_arr=("S30" "S52")
# # subj_arr=("S56" "S67")


bids_root="~/cworkspace/BIDS_1.0_Sentence_µECoG/BIDS"
task="sentence"
subj_arr=("S91")


features="[high_gamma,spike_band]"
# features="[high_gamma]"
# features="[spike_band]"
# features="[delta,theta,alpha,beta,low_gamma]"
# task_period="[perception,production]"
# task_period="perception"
task_period="production"
# task_period="delay"

useSig=true  # leave this lower case
# useSig=false  # leave this lower case
sig_channels_method="tpc"

derivative_name=null
# derivative_name="phonemeLevel"
# derivative_name="wordLevel"

# Spatial-average mode: when spatialAvg=true, this script submits one job
# per contact size in spatialAvgContactSizes; each job loads the matching
# derivatives/spatialAvg/sub-XX_..._desc-contact{N}_ieeg.edf produced by
# spatial_avg_raw.py and writes epochs to
# derivatives/epoch(spatialAvgContact{N})({reference})/.
# spatial_avg.enabled and derivative_name are mutually exclusive in
# extract_ieeg_epochs.py (spatial_avg takes priority for raw selection).
spatialAvg=false  # leave this lower case
# spatialAvg=true
spatialAvgContactSizes=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# spatialAvgContactSizes=(2)

mem="128G"
if [ "${derivative_name}" == "wordLevel" ]; then
    mem="256G"
elif [ "${derivative_name}" == "phonemeLevel" ]; then
    mem="512G"
fi


for subject in "${subj_arr[@]}"; do
    if [[ "${spatialAvg}" == "true" ]]; then
        for contact_size in "${spatialAvgContactSizes[@]}"; do
            echo "----------------------------------------"
            echo "Creating spatial-avg processing job for ${subject} (contact_size=${contact_size})"
            job_name="${subject}_extract_ieeg_epochs_sig${useSig}_SA${contact_size}"
            echo "Job name: ${job_name}"
            sbatch -J "${job_name}" --mem="${mem}" run_extract_ieeg_epochs.sh \
                -s "${subject}" -b "${bids_root}" -t "${task}" \
                -f "${features}" -p "${task_period}" \
                -u "${useSig}" -d "${derivative_name}" \
                -a true -z "${contact_size}"
            echo "----------------------------------------"
        done
    else
        echo "----------------------------------------"
        echo "Creating processing job for ${subject}"
        if [[ "${derivative_name}" = "null" ]]; then
            deriv_suffix=""
        else
            deriv_suffix="_${derivative_name}"
        fi
        job_name="${subject}_extract_ieeg_epochs_sig${useSig}${deriv_suffix}"
        echo "Job name: ${job_name}"
        sbatch -J "${job_name}" --mem="${mem}" run_extract_ieeg_epochs.sh \
            -s "${subject}" -b "${bids_root}" -t "${task}" \
            -f "${features}" -p "${task_period}" \
            -u "${useSig}" -d "${derivative_name}" \
            -a false
        echo "----------------------------------------"
    fi
done