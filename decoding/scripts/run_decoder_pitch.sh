#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/decode_pitchSubsample/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/decode_pitchSubsample/%j.error
#SBATCH -p common,scavenger
#SBATCH -c 20
#SBATCH --mem=40G

# ----------------------------
# Load environment
# ----------------------------
source ~/.bashrc
conda activate ieeg

# ----------------------------
# Arguments
# ----------------------------
SUBJECT=""
BIDS_ROOT="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives"
LAYOUT_ROOT="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
TASK="lexical"
PHONEME_IDX=0
N_PHONS=5
N_FOLDS=20
N_SUBSAMPLE_ITERS=50
TW_MIN=-0.5
TW_MAX=0.5
DESCRIPTION="productionMeanSub"
SUFFIX="highgamma"
EXTENSION=".fif"
DATATYPE="epoch(band)(power)"
CHANCE=false
PITCH_MM=0.4
SEED=""

while getopts s:b:l:t:p:n:f:i:w:x:d:u:e:a:c:P:r: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        l) LAYOUT_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        p) PHONEME_IDX=${OPTARG};;
        n) N_PHONS=${OPTARG};;
        f) N_FOLDS=${OPTARG};;
        i) N_SUBSAMPLE_ITERS=${OPTARG};;
        w) TW_MIN=${OPTARG};;
        x) TW_MAX=${OPTARG};;
        d) DESCRIPTION=${OPTARG};;
        u) SUFFIX=${OPTARG};;
        e) EXTENSION=${OPTARG};;
        a) DATATYPE=${OPTARG};;
        c) CHANCE=${OPTARG};;
        P) PITCH_MM=${OPTARG};;
        r) SEED=${OPTARG};;
    esac
done

if [[ -z "${SUBJECT}" ]]; then
    echo "Please specify patient id with -s"
    exit 1
fi

# ----------------------------
# Run script with Hydra config
# ----------------------------
cd ..
echo "Pitch-subsampling decode for subject ${SUBJECT}, pitch_mm=${PITCH_MM}"

SEED_ARG=""
if [[ -n "${SEED}" ]]; then
    SEED_ARG="seed=${SEED}"
fi

python decode_bids_phonemes_pitch.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    "layout_root='${LAYOUT_ROOT}'" \
    task=${TASK} \
    phoneme_idx=${PHONEME_IDX} \
    n_phons=${N_PHONS} \
    n_folds=${N_FOLDS} \
    n_subsample_iters=${N_SUBSAMPLE_ITERS} \
    "time_window=[${TW_MIN},${TW_MAX}]" \
    description=${DESCRIPTION} \
    suffix=${SUFFIX} \
    extension=${EXTENSION} \
    "datatype='${DATATYPE}'" \
    compute_chance=${CHANCE} \
    pitch_mm=${PITCH_MM} \
    ${SEED_ARG} \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
