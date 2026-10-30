#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/decode_crossPtTask/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/decode_crossPtTask/%j.error
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 20
#SBATCH --mem=200G

# ----------------------------
# Load environment
# ----------------------------
source ~/.bashrc
conda activate ieeg

# ----------------------------
# Arguments
# ----------------------------
SUBJECT=""
BIDS_ROOT="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
TASK="lexical"
PHONEME_IDX=-1
POOL_TASK="all"
N_FOLDS=20
N_ITER=10
TW_MIN=""
TW_MAX=""
DESCRIPTION="productionMeanSub"
SUFFIX="highgamma"
EXTENSION=".fif"
DATATYPE="epoch(band)(power)"
OUTPUT_SUFFIX=""

while getopts s:b:t:p:o:f:i:w:x:n:d:u:e:a:h: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        p) PHONEME_IDX=${OPTARG};;
        o) POOL_TASK=${OPTARG};;
        f) N_FOLDS=${OPTARG};;
        i) N_ITER=${OPTARG};;
        w) TW_MIN=${OPTARG};;
        x) TW_MAX=${OPTARG};;
        n) N_PHONS=${OPTARG};;
        d) DESCRIPTION=${OPTARG};;
        u) SUFFIX=${OPTARG};;
        e) EXTENSION=${OPTARG};;
        a) DATATYPE=${OPTARG};;
        h) OUTPUT_SUFFIX=${OPTARG};;
    esac
done

if [[ -z "${SUBJECT}" ]]; then
    echo "Please specify patient id with -s"
    exit 1
fi

# Build time_window argument
if [[ -n "${TW_MIN}" && -n "${TW_MAX}" ]]; then
    TW_ARG="time_window=[${TW_MIN},${TW_MAX}]"
else
    TW_ARG="time_window=null"
fi

# ----------------------------
# Run script with Hydra config
# ----------------------------
cd ..
echo "Cross-patient decoding: subject ${SUBJECT}, pool_task ${POOL_TASK}, phoneme_idx ${PHONEME_IDX}"
python decode_crossPt_phonemes.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    task=${TASK} \
    phoneme_idx=${PHONEME_IDX} \
    pool_task=${POOL_TASK} \
    n_phons=${N_PHONS} \
    n_folds=${N_FOLDS} \
    n_iter=${N_ITER} \
    "${TW_ARG}" \
    description=${DESCRIPTION} \
    suffix=${SUFFIX} \
    extension=${EXTENSION} \
    "datatype='${DATATYPE}'" \
    output_suffix=${OUTPUT_SUFFIX} \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
