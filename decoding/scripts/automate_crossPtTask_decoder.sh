#!/bin/bash
subj_arr=("S81")
# subj_arr=("S41" "S45" "S47" "S51" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78")
# subj_arr=("S41" "S45" "S47" "S51" "S53" "S55" "S56" "S63" "S67" "S74" "S75" "S76")


bids_root="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
phonemeIdx=-1
poolTask="lexical"
nFolds=20
nIter=10
# tw=(-0.5 0.5)
tw=(-1 1) #tw for alignment, decoder is set to -0.5 to 0.5 in script
description="productionMeanSub"
suffix="highgamma"
ext=".fif"
task="lexical"
datatype="epoch(band)(power)(sig)"

for subject in "${subj_arr[@]}"; do
    echo "----------------------------------------"
    echo "Creating processing job for ${subject}, phonemeIdx ${phonemeIdx}, poolTask ${poolTask}"
    job_name="${subject}_${phonemeIdx}_${poolTask}_decode_crossPtTask"
    sbatch -J $job_name run_crossPtTask_decoder.sh -b ${bids_root} -s ${subject} -p ${phonemeIdx} -o ${poolTask} -f ${nFolds} -i ${nIter} -w ${tw[0]} -x ${tw[1]} -d ${description} -u ${suffix} -e ${ext} -t ${task} -a ${datatype}
    echo "----------------------------------------"
done