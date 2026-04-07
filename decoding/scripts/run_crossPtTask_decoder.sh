#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/decode_crossPtTask/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/decode_crossPtTask/%j.error
#SBATCH -p common,scavenger
#SBATCH -c 20
#SBATCH --mem=64G

source ~/.bashrc
conda activate ieeg


bids_root="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
subject=""
phonemeIdx=-1
poolTask="all"
nFolds=20
nIter=10
tw=(None None)
description="productionMeanSub"
suffix="highgamma"
ext=".fif"
task="lexical"
datatype="epoch(band)(power)"
while getopts b:s:p:o:f:i:w:x:d:u:e:t:a: flag
do
    case "${flag}" in
        b) bids_root=${OPTARG};;
        s) subject=${OPTARG};;
        p) phonemeIdx=${OPTARG};;
        o) poolTask=${OPTARG};;
        f) nFolds=${OPTARG};;
        i) nIter=${OPTARG};;
        w) tmin=(${OPTARG});;
        x) tmax=(${OPTARG});;
        d) description=${OPTARG};;
        u) suffix=${OPTARG};;
        e) ext=${OPTARG};;
        t) task=${OPTARG};;
        a) datatype=${OPTARG};;
    esac
done
tw=(${tmin} ${tmax})

if [[ -z "${subject}" ]]; then
    echo "Please specify patient id with -s"
    exit 1
fi

cd ..
echo "Running decoder for subject ${subject} with cross-pt task ${poolTask} phonemeIdx ${phonemeIdx}, nFolds ${nFolds}, nIter ${nIter}, tw ${tw[0]} ${tw[1]}, description ${description}, suffix ${suffix}, ext ${ext}, task ${task}, datatype ${datatype}"
python decode_bids_crossPatientTask_phonemes.py ${bids_root} ${subject} ${phonemeIdx} ${poolTask} --nFolds ${nFolds} --nIter ${nIter} --tw ${tw[0]} ${tw[1]} --description ${description} --suffix ${suffix} --ext ${ext} --task ${task} --datatype ${datatype}
cd scripts