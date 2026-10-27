#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/extract_epochs/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/extract_epochs/%j.error
#SBATCH -p common,scavenger,coganlab-gpu
#SBATCH -c 25
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# ----------------------------
# Load environment
# ----------------------------
source ~/.bashrc
conda activate ieeg

# ---------------------------
# Arguments
# ----------------------------
SUBJECT=""
TASK="lexical"
BIDS_ROOT="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
FEATURES="high_gamma"
TASK_PERIOD="production"
USE_SIG=false
DERIV_NAME=null
# Spatial-average mode: when SA_ENABLED=true, the raw input is read from
# derivatives/spatialAvg/ for the requested SA_CONTACT_SIZE (produced by
# spatial_avg_raw.py).  SA_CONTACT_SIZE must be set when SA_ENABLED=true.
SA_ENABLED=false
SA_CONTACT_SIZE=""

while getopts s:b:t:f:p:u:d:a:z: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        f) FEATURES=${OPTARG};;
        p) TASK_PERIOD=${OPTARG};;
        u) USE_SIG=${OPTARG};;
        d) DERIV_NAME=${OPTARG};;
        a) SA_ENABLED=${OPTARG};;
        z) SA_CONTACT_SIZE=${OPTARG};;
    esac
done

if [[ -z "${SUBJECT}" ]]; then
    echo "Please specify patient id with -s"
    exit 1
fi

if [[ "${SA_ENABLED}" == "true" && -z "${SA_CONTACT_SIZE}" ]]; then
    echo "spatial_avg.enabled=true requires -z <contact_size>"
    exit 1
fi

# ----------------------------
# Run script with Hydra config
# ----------------------------
# cd ../..
cd ..
echo "Extracting epochs for subject ${SUBJECT} with features ${FEATURES} and task period ${TASK_PERIOD} (derivative_name=${DERIV_NAME}, spatial_avg.enabled=${SA_ENABLED}, spatial_avg.contact_size=${SA_CONTACT_SIZE:-null})"

SA_ARGS="spatial_avg.enabled=${SA_ENABLED}"
if [[ -n "${SA_CONTACT_SIZE}" ]]; then
    SA_ARGS="${SA_ARGS} spatial_avg.contact_size=${SA_CONTACT_SIZE}"
fi

# python preprocessing/extract_ieeg_epochs.py \
python extract_ieeg_epochs.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    task=${TASK} \
    features=${FEATURES} \
    task_periods=${TASK_PERIOD} \
    sig_channels=${USE_SIG} \
    derivative_name=${DERIV_NAME} \
    ${SA_ARGS} \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}