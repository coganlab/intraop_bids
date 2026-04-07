#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/extract_epochs/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/extract_epochs/%j.error
#SBATCH -p common,scavenger
#SBATCH -c 25
#SBATCH --mem=128G

# ----------------------------
# Load environment
# ----------------------------
source ~/.bashrc
conda activate ieeg

# ---------------------------
# Arguments
# ----------------------------
SUBJECT=""
BIDS_ROOT="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
FEATURES="high_gamma"
TASK_PERIOD="production"
USE_SIG=false

while getopts s:b:f:t:u: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        f) FEATURES=${OPTARG};;
        t) TASK_PERIOD=${OPTARG};;
        u) USE_SIG=${OPTARG};;
    esac
done

if [[ -z "${SUBJECT}" ]]; then
    echo "Please specify patient id with -s"
    exit 1
fi

# ----------------------------
# Run script with Hydra config
# ----------------------------
# cd ../..
cd ..
echo "Extracting epochs for subject ${SUBJECT} with features ${FEATURES} and task period ${TASK_PERIOD}"
# python preprocessing/extract_ieeg_epochs.py \
python extract_ieeg_epochs.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    features=${FEATURES} \
    task_periods=${TASK_PERIOD} \
    sig_channels=${USE_SIG} \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}