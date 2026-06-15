#!/bin/bash

subj_arr=("S83")
# subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81" "S82" "S83")

bids_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
layout_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
phonemeIdx=(-1 1 2 3 4 5)
nPhons=5

nFolds=10
extraIters=1
chance=false
twMin=-0.5
twMax=0.5

# task_periods=("perception" "production")
task_periods=("production")
norm_type="MeanSub"
suffix="highGamma"
extension=".fif"
task="lexical"
# datatype="epoch(band)(power)(sig)"
datatype="epoch(band)(power)"

# Square subgrid sizes to sweep over (rows == cols here, but the script
# accepts independent dims via -R and -C).
win_sizes=("3 6" "4 8" "5 10" "6 12" "7 14" "8 16" "9 18" "10 20" "11 22" "12 24" "13 26" "14 28" "15 30" "16 32")
# win_sizes=("2 4")

# Compact the chanmap (drop fully-NaN rows/columns) before enumerating
# valid subgrid placements.  Recommended for multi-array recordings
# like S83 where a NaN gap row between two arrays would otherwise block
# larger windows from fitting.
compact_chanmap="true"


for subject in "${subj_arr[@]}"; do
    for task_period in "${task_periods[@]}"; do
        description="${task_period}${norm_type}"
        for phon in "${phonemeIdx[@]}"; do
            for win in "${win_sizes[@]}"; do
                read -r wr wc <<< "${win}"
                echo "----------------------------------------"
                echo "Job: ${subject} ${task_period} phon=${phon} win=${wr}x${wc} compact=${compact_chanmap}"
                job_name="${subject}_${task_period}_p${phon}_win${wr}x${wc}_decode_coverageSubsample"
                sbatch -J $job_name run_decoder_coverage.sh \
                    -s ${subject} -b "${bids_root}" -l "${layout_root}" -t ${task} \
                    -p ${phon} -n ${nPhons} -f ${nFolds} -i ${extraIters} \
                    -w ${twMin} -x ${twMax} \
                    -d ${description} -u ${suffix} -e ${extension} \
                    -a "${datatype}" -c ${chance} -R ${wr} -C ${wc} \
                    -K ${compact_chanmap}
                echo "----------------------------------------"
            done
        done
    done
done
