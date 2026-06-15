#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/decode_phonemeLevel/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/decode_phonemeLevel/%j.error
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH --time=01:00:00
#SBATCH -c 20
#SBATCH --mem=128G

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
TASK="lexical"
PHONEME_IDX=0
N_PHONS=5
N_FOLDS=20
N_ITER=50
TW_MIN=-0.5
TW_MAX=0.5
DESCRIPTION="productionMeanSub"
SUFFIX="highgamma"
EXTENSION=".fif"
DATATYPE="epoch(band)(power)"
CHANCE=false

while getopts s:b:t:p:n:f:i:w:x:d:u:e:a:c: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        p) PHONEME_IDX=${OPTARG};;
        n) N_PHONS=${OPTARG};;
        f) N_FOLDS=${OPTARG};;
        i) N_ITER=${OPTARG};;
        w) TW_MIN=${OPTARG};;
        x) TW_MAX=${OPTARG};;
        d) DESCRIPTION=${OPTARG};;
        u) SUFFIX=${OPTARG};;
        e) EXTENSION=${OPTARG};;
        a) DATATYPE=${OPTARG};;
        c) CHANCE=${OPTARG};;
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
echo "Decoding subject ${SUBJECT}, phoneme_idx ${PHONEME_IDX}, suffix ${SUFFIX}, description ${DESCRIPTION}"
python decode_bids_phonemes.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    task=${TASK} \
    phoneme_idx=${PHONEME_IDX} \
    n_phons=${N_PHONS} \
    n_folds=${N_FOLDS} \
    n_iter=${N_ITER} \
    "time_window=[${TW_MIN},${TW_MAX}]" \
    description=${DESCRIPTION} \
    suffix=${SUFFIX} \
    extension=${EXTENSION} \
    "datatype='${DATATYPE}'" \
    compute_chance=${CHANCE} \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
