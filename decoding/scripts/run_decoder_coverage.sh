#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/decode_coverageSubsample/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/decode_coverageSubsample/%j.error
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
EXTRA_ITERS=1
TW_MIN=-0.5
TW_MAX=0.5
DESCRIPTION="productionMeanSub"
SUFFIX="highgamma"
EXTENSION=".fif"
DATATYPE="epoch(band)(power)"
CHANCE=false
WIN_ROWS=4
WIN_COLS=4

while getopts s:b:l:t:p:n:f:i:w:x:d:u:e:a:c:R:C: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        l) LAYOUT_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        p) PHONEME_IDX=${OPTARG};;
        n) N_PHONS=${OPTARG};;
        f) N_FOLDS=${OPTARG};;
        i) EXTRA_ITERS=${OPTARG};;
        w) TW_MIN=${OPTARG};;
        x) TW_MAX=${OPTARG};;
        d) DESCRIPTION=${OPTARG};;
        u) SUFFIX=${OPTARG};;
        e) EXTENSION=${OPTARG};;
        a) DATATYPE=${OPTARG};;
        c) CHANCE=${OPTARG};;
        R) WIN_ROWS=${OPTARG};;
        C) WIN_COLS=${OPTARG};;
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
echo "Coverage-subsample decode for ${SUBJECT}, win=${WIN_ROWS}x${WIN_COLS}"

python decode_bids_phonemes_coverage.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    "layout_root='${LAYOUT_ROOT}'" \
    task=${TASK} \
    phoneme_idx=${PHONEME_IDX} \
    n_phons=${N_PHONS} \
    n_folds=${N_FOLDS} \
    extra_iters_per_subgrid=${EXTRA_ITERS} \
    "time_window=[${TW_MIN},${TW_MAX}]" \
    description=${DESCRIPTION} \
    suffix=${SUFFIX} \
    extension=${EXTENSION} \
    "datatype='${DATATYPE}'" \
    compute_chance=${CHANCE} \
    "win_size=[${WIN_ROWS},${WIN_COLS}]" \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
