#!/bin/bash
# subj_arr=("S81")
subj_arr=("S41" "S45" "S47" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78")
bids_root="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
task="lexical"
phonemeIdx=-1

poolTask="lexical"
nFolds=20
nIter=10
twMin=-1
twMax=1

task_periods=("production" "perception")
norm_type="MeanSub"
# norm_type="Zscore"
suffix="highGamma"
# suffix="spikeBand"
extension=".fif"

datatype="epoch(band)(power)(sig)"

for subject in "${subj_arr[@]}"; do
    for task_period in "${task_periods[@]}"; do
        description="${task_period}${norm_type}"
        for phon in "${phonemeIdx[@]}"; do
            echo "----------------------------------------"
            echo "Creating processing job for ${subject}, task_period ${task_period}, phonemeIdx ${phon}"
            job_name="${subject}_${task_period}_${phon}_${poolTask}_decode_crossPtTask"
            echo $job_name
            sbatch -J $job_name run_crosspt_decoder.sh \
                -s ${subject} -b "${bids_root}" -t ${task} \
                -p ${phon} -o ${poolTask} -f ${nFolds} -i ${nIter} \
                -w ${twMin} -x ${twMax} \
                -d ${description} -u ${suffix} -e ${extension} \
                -a "${datatype}"
            echo "----------------------------------------"
        done
    done
done
