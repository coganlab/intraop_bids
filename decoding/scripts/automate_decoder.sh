#!/bin/bash

# # subj_arr=("S14")
# subj_arr=("S14" "S16" "S18" "S19" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")
# # subj_arr=("S16" "S18" "S19" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")
# bids_root="~/cworkspace/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
# phonemeIdx=(-1 1 2 3)
# # phonemeIdx=(1 2 3)
# # phonemeIdx=(-1)
# nPhons=3
# task="phoneme"

# subj_arr=("S86")
# # subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81" "S82" "S83")
# bids_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
# phonemeIdx=(-1 1 2 3 4 5)
# # phonemeIdx=(1 2 3 4 5)
# # phonemeIdx=(-1)
# # phonemeIdx=(-1 1)
# nPhons=5
# task="lexical"

subj_arr=("S91")
bids_root="~/cworkspace/BIDS_1.0_Sentence_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
phonemeIdx=(-1)
nPhons=1 # sentence decoding only works with phonemeIdx=-1 (i.e., all phonemes) due to variable phoneme number so nPhons doesn't matter
task="sentence"

nFolds=10
nIter=50
chance=true
# chance=false
twMin=-0.5
twMax=0.5

task_periods=("perception" "production")
# task_periods=("production")
norm_type="MeanSub"
# description="Zscore"
suffix="highGamma"
# suffix="spikeBand"
# suffix='[highGamma,spikeBand]'
# suffix='lowGamma'
# suffix='beta'
# suffix='alpha'
# suffix='theta'
# suffix='delta'
# suffix='[highGamma,lowGamma]'
# suffix='[highGamma,beta]'
# suffix='[highGamma,alpha]'
# suffix='[highGamma,theta]'
# suffix='[highGamma,delta]'
extension=".fif"

# datatype="epoch(band)(power)(sig)"
datatype="epoch(band)(power)"

for subject in "${subj_arr[@]}"; do
    for task_period in "${task_periods[@]}"; do
        description="${task_period}${norm_type}"
        for phon in "${phonemeIdx[@]}"; do
            echo "----------------------------------------"
            echo "Creating processing job for ${subject}, task_period ${task_period}, phonemeIdx ${phon}"
            job_name="${subject}_${task_period}_${phon}_decode_phonemeLevel"
            echo $job_name
            sbatch -J $job_name run_decoder.sh \
                -s ${subject} -b "${bids_root}" -t ${task} \
                -p ${phon} -n ${nPhons} -f ${nFolds} -i ${nIter} \
                -w ${twMin} -x ${twMax} \
                -d ${description} -u ${suffix} -e ${extension} \
                -a "${datatype}" -c ${chance}
            echo "----------------------------------------"
        done
    done
done
