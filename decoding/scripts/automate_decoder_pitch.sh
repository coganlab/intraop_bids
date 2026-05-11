#!/bin/bash

# subj_arr=("S82")
subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81" "S82" "S83")

bids_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
layout_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
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

# Pitches in mm to sweep over.
pitches=(0.4 0.6 0.8 1.2 1.6 2.4)


for subject in "${subj_arr[@]}"; do
    for task_period in "${task_periods[@]}"; do
        description="${task_period}${norm_type}"
        for phon in "${phonemeIdx[@]}"; do
            for pitch in "${pitches[@]}"; do
                echo "----------------------------------------"
                echo "Job: ${subject} ${task_period} phon=${phon} pitch=${pitch}"
                job_name="${subject}_${task_period}_p${phon}_pitch${pitch}_decode_pitchSubsample"
                sbatch -J $job_name run_decoder_pitch.sh \
                    -s ${subject} -b "${bids_root}" -l "${layout_root}" -t ${task} \
                    -p ${phon} -n ${nPhons} -f ${nFolds} -i ${nSubsampleIters} \
                    -w ${twMin} -x ${twMax} \
                    -d ${description} -u ${suffix} -e ${extension} \
                    -a "${datatype}" -c ${chance} -P ${pitch}
                echo "----------------------------------------"
            done
        done
    done
done
