#!/bin/bash

# subj_arr=("S82")
subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81" "S82" "S83")

bids_root_parent="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives"
phonemeIdx=(-1 1 2 3 4 5)
nPhons=5

nFolds=10
nSubsampleIters=50
chance=false
twMin=-0.5
twMax=0.5

task_periods=("perception" "production")
norm_type="MeanSub"
suffix="spikeBand"
extension=".fif"
task="lexical"
datatype="epoch(band)(power)(sig)"
reference="CAR"

# Contact sizes (must match what spatial_avg_raw.py + extract_ieeg_epochs.py
# produced).  Each value loads epochs from
# .../derivatives/epoch(spatialAvgContact{N})(CAR)/
contact_sizes=(2 4 8)


for subject in "${subj_arr[@]}"; do
    for task_period in "${task_periods[@]}"; do
        description="${task_period}${norm_type}"
        for phon in "${phonemeIdx[@]}"; do
            for cs in "${contact_sizes[@]}"; do
                echo "----------------------------------------"
                echo "Job: ${subject} ${task_period} phon=${phon} contact=${cs}"
                job_name="${subject}_${task_period}_p${phon}_contact${cs}_decode_spatialAvgSubsample"
                sbatch -J $job_name run_decoder_spatial_avg.sh \
                    -s ${subject} -B "${bids_root_parent}" -t ${task} \
                    -p ${phon} -n ${nPhons} -f ${nFolds} -i ${nSubsampleIters} \
                    -w ${twMin} -x ${twMax} \
                    -d ${description} -u ${suffix} -e ${extension} \
                    -a "${datatype}" -c ${chance} -Z ${cs} -r ${reference}
                echo "----------------------------------------"
            done
        done
    done
done
