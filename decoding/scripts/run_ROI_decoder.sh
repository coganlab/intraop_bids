#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/decode_ROI/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/decode_ROI/%j.error
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 20
#SBATCH --mem=256G

# ----------------------------
# Load environment
# ----------------------------
source ~/.bashrc
conda activate ieeg

# ----------------------------
# Arguments
# ----------------------------
ROI=""
SUBJECTs=""
BIDS_ROOT="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
TASK="lexical"
PHONEME_IDX=-1
N_FOLDS=20
N_ITER=10
TW_MIN=""
TW_MAX=""
DESCRIPTION="productionZscore"
SUFFIX="highGamma"
EXTENSION=".fif"
DATATYPE="epoch(band)(power)"
CHANCE=false

while getopts r:s:b:t:p:f:i:w:x:n:d:u:e:a:c: flag
do
    case "${flag}" in
        r) ROI=${OPTARG};;
        s) SUBJECTS=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        p) PHONEME_IDX=${OPTARG};;
        f) N_FOLDS=${OPTARG};;
        i) N_ITER=${OPTARG};;
        w) TW_MIN=${OPTARG};;
        x) TW_MAX=${OPTARG};;
        n) N_PHONS=${OPTARG};;
        d) DESCRIPTION=${OPTARG};;
        u) SUFFIX=${OPTARG};;
        e) EXTENSION=${OPTARG};;
        a) DATATYPE=${OPTARG};;
        c) CHANCE=${OPTARG};;
    esac
done

if [[ -z "${ROI}" ]]; then
    echo "Please specify ROI with -r"
    exit 1
fi

if [[ -z "${SUBJECTS}" ]]; then
    echo "Please specify subject(s) with -s"
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
echo "ROI decoding: ROI ${ROI}, subject(s) ${SUBJECTS}, pool_task ${POOL_TASK}, phoneme_idx ${PHONEME_IDX}"
python decode_ROI_phonemes.py \
    roi=${ROI} \
    "patients=[${SUBJECTS}]" \
    "bids_root='${BIDS_ROOT}'" \
    task=${TASK} \
    phoneme_idx=${PHONEME_IDX} \
    n_phons=${N_PHONS} \
    n_folds=${N_FOLDS} \
    n_iter=${N_ITER} \
    compute_chance=${CHANCE} \
    "${TW_ARG}" \
    description=${DESCRIPTION} \
    suffix=${SUFFIX} \
    extension=${EXTENSION} \
    "datatype='${DATATYPE}'" \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
