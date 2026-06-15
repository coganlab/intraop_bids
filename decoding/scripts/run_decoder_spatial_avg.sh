#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/decode_spatialAvgSubsample/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/decode_spatialAvgSubsample/%j.error
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
# Spatial-avg epoch derivative root (computed automatically below from
# BIDS_ROOT_PARENT + contact size if BIDS_ROOT is left blank).
BIDS_ROOT=""
BIDS_ROOT_PARENT="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives"
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
CONTACT_SIZE=2
REFERENCE="CAR"

while getopts s:b:B:t:p:n:f:i:w:x:d:u:e:a:c:Z:r: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        B) BIDS_ROOT_PARENT=${OPTARG};;
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
        Z) CONTACT_SIZE=${OPTARG};;
        r) REFERENCE=${OPTARG};;
    esac
done

if [[ -z "${SUBJECT}" ]]; then
    echo "Please specify patient id with -s"
    exit 1
fi

# If BIDS_ROOT not explicitly set, derive it from BIDS_ROOT_PARENT +
# contact size + reference (mirroring the derivative path written by
# extract_ieeg_epochs.py with spatial_avg.enabled=true).
if [[ -z "${BIDS_ROOT}" ]]; then
    BIDS_ROOT="${BIDS_ROOT_PARENT}/epoch(spatialAvgContact${CONTACT_SIZE})(${REFERENCE})"
fi

# ----------------------------
# Run script with Hydra config
# ----------------------------
cd ..
echo "Spatial-avg decode for ${SUBJECT}, contact_size=${CONTACT_SIZE}"
echo "BIDS root: ${BIDS_ROOT}"

python decode_bids_phonemes_spatialAvg.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
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
    contact_size=${CONTACT_SIZE} \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
