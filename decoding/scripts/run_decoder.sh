#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/decode_phonemeLevel/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/decode_phonemeLevel/%j.error
#SBATCH -p common,scavenger
#SBATCH -c 20
#SBATCH --mem=40G

source ~/.bashrc
conda activate ieeg


bids_root="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives"
subject=""
phonemeIdx=0
nPhons=5
nFolds=20
nIter=50
tw=(-0.5 0.5)
description="productionMeanSub"
suffix="highgamma"
ext=".fif"
task="PhonemeSequence"
datatype="epoch(band)(power)"
chance=false

while getopts b:s:p:n:f:i:w:x:d:u:e:t:a:c: flag
do
    case "${flag}" in
        b) bids_root=${OPTARG};;
        s) subject=${OPTARG};;
        p) phonemeIdx=${OPTARG};;
        n) nPhons=${OPTARG};;
        f) nFolds=${OPTARG};;
        i) nIter=${OPTARG};;
        w) tmin=(${OPTARG});;
        x) tmax=(${OPTARG});;
        d) description=${OPTARG};;
        u) suffix=${OPTARG};;
        e) ext=${OPTARG};;
        t) task=${OPTARG};;
        a) datatype=${OPTARG};;
        c) chance=${OPTARG};;
    esac
done
tw=(${tmin} ${tmax})

if [[ -z "${subject}" ]]; then
    echo "Please specify patient id with -s"
    exit 1
fi

cd ..
echo "Running decoder for subject ${subject} with phonemeIdx ${phonemeIdx}, nPhons ${nPhons}, nFolds ${nFolds}, nIter ${nIter}, tw ${tw[0]} ${tw[1]}, description ${description}, suffix ${suffix}, ext ${ext}, task ${task}, datatype ${datatype}, chance ${chance}"
python decode_bids_phonemes.py ${bids_root} ${subject} ${phonemeIdx} --nPhons ${nPhons} --nFolds ${nFolds} --nIter ${nIter} --tw ${tw[0]} ${tw[1]} --description ${description} --suffix ${suffix} --ext ${ext} --task ${task} --datatype ${datatype} --chance ${chance}
cd scripts