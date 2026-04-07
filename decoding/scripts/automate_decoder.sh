#!/bin/bash
# subj_arr=("S41")
subj_arr=("S41" "S45" "S47" "S51" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81")


bids_root="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
phonemeIdx=(-1 1 2 3 4 5)
nPhons=5
nFolds=10
nIter=50
# chance=true
chance=false
tw=(-0.5 0.5)
description="productionMeanSub"
# description="productionZscore"
# suffix="highgamma"
suffix="spikeBand"
ext=".fif"
task="lexical"
datatype="epoch(band)(power)(sig)"
# datatype="epoch(band)(power)"

for subject in "${subj_arr[@]}"; do
    for phon in "${phonemeIdx[@]}"; do
        echo "----------------------------------------"
        echo "Creating processing job for ${subject}, phonemeIdx ${phon}"
        job_name="${subject}_${phon}_decode_phonemeLevel"
        echo $job_name
        sbatch -J $job_name run_decoder.sh -b ${bids_root} -s ${subject} -p ${phon} -n ${nPhons} -f ${nFolds} -i ${nIter} -w ${tw[0]} -x ${tw[1]} -d ${description} -u ${suffix} -e ${ext} -t ${task} -a ${datatype} -c ${chance}
        echo "----------------------------------------"
    done
done