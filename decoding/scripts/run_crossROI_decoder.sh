#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/decode_crossROI/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/decode_crossROI/%j.error
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 20
#SBATCH --mem=64G

# ----------------------------
# Load environment
# ----------------------------
source ~/.bashrc
conda activate ieeg

# ----------------------------
# Arguments
# ----------------------------
TAR_ROI=""
TAR_SUBJECTs=""
SRC_ROI=""
SRC_SUBJECTs=""
BIDS_ROOT="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
TASK="lexical"
PHONEME_IDX=-1
POOL_TASK="all"
N_FOLDS=20
N_ITER=10
TW_MIN=""
TW_MAX=""
DESCRIPTION="productionZscore"
SUFFIX="highGamma"
EXTENSION=".fif"
DATATYPE="epoch(band)(power)"

while getopts r:s:c:j:b:t:p:o:f:i:w:x:n:d:u:e:a: flag
do
    case "${flag}" in
        r) TAR_ROI=${OPTARG};;
        s) TAR_SUBJECTs=${OPTARG};;
        c) SRC_ROI=${OPTARG};;
        j) SRC_SUBJECTs=${OPTARG};;
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
    esac
done

if [[ -z "${TAR_ROI}" ]]; then
    echo "Please specify target ROI with -r"
    exit 1
fi


if [[ -z "${TAR_SUBJECTs}" ]]; then
    echo "Please specify target subject(s) with -s"
    exit 1
fi

if [[ -z "${SRC_ROI}" ]]; then
    echo "Please specify source ROI with -c"
    exit 1
fi

if [[ -z "${SRC_SUBJECTs}" ]]; then
    echo "Please specify source subject(s) with -j"
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
echo "Cross-ROI decoding: Target ROI ${TAR_ROI}, Target subject(s) ${TAR_SUBJECTs}, Source ROI ${SRC_ROI}, Source subject(s) ${SRC_SUBJECTs}, pool_task ${POOL_TASK}, phoneme_idx ${PHONEME_IDX}"
python decode_crossROI_phonemes.py \
    target_rois=${TAR_ROI} \
    "target_patients=[${TAR_SUBJECTs}]" \
    "source_rois=[${SRC_ROI}]" \
    "source_patients=[${SRC_SUBJECTs}]" \
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
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
