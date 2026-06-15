#!/bin/bash
# bids_root="~/cworkspace/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS"
# task="phoneme"
# subj_arr=("S14" "S16" "S18" "S19" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")
# # subj_arr=("S14")


bids_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
task="lexical"
subj_arr=("S86")
# subj_arr=("S30" "S82" "S83")
# subj_arr=("S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81")
# subj_arr=("S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81")
# subj_arr=("S30" "S52")
# subj_arr=("S56" "S67")


features="[high_gamma,spike_band]"
# features="[high_gamma]"
# features="[spike_band]"
# features="[delta,theta,alpha,beta,low_gamma]"
# task_period="[perception,production]"
# task_period="perception"
task_period="production"

useSig=true  # leave this lower case
# useSig=false  # leave this lower case

# phonemeLevel=true  # leave this lower case
phonemeLevel=false  # leave this lower case

# Spatial-average mode: when spatialAvg=true, this script submits one job
# per contact size in spatialAvgContactSizes; each job loads the matching
# derivatives/spatialAvg/sub-XX_..._desc-contact{N}_ieeg.edf produced by
# spatial_avg_raw.py and writes epochs to
# derivatives/epoch(spatialAvgContact{N})({reference})/.
# spatial_avg.enabled and phoneme_level are mutually exclusive in
# extract_ieeg_epochs.py (spatial_avg takes priority for raw selection).
spatialAvg=false  # leave this lower case
# spatialAvg=true
spatialAvgContactSizes=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# spatialAvgContactSizes=(2)


for subject in "${subj_arr[@]}"; do
    if [[ "${spatialAvg}" == "true" ]]; then
        for contact_size in "${spatialAvgContactSizes[@]}"; do
            echo "----------------------------------------"
            echo "Creating spatial-avg processing job for ${subject} (contact_size=${contact_size})"
            job_name="${subject}_extract_ieeg_epochs_sig${useSig}_PL${phonemeLevel}_SA${contact_size}"
            sbatch -J "${job_name}" run_extract_ieeg_epochs.sh \
                -s "${subject}" -b "${bids_root}" -t "${task}" \
                -f "${features}" -p "${task_period}" \
                -u "${useSig}" -l "${phonemeLevel}" \
                -a true -z "${contact_size}"
            echo "----------------------------------------"
        done
    else
        echo "----------------------------------------"
        echo "Creating processing job for ${subject}"
        job_name="${subject}_extract_ieeg_epochs_sig${useSig}_PL${phonemeLevel}"
        sbatch -J "${job_name}" run_extract_ieeg_epochs.sh \
            -s "${subject}" -b "${bids_root}" -t "${task}" \
            -f "${features}" -p "${task_period}" \
            -u "${useSig}" -l "${phonemeLevel}" \
            -a false
        echo "----------------------------------------"
    fi
done