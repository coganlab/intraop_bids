#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/specChanMap/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/specChanMap/%j.error
#SBATCH -p common,scavenger
#SBATCH -c 25
#SBATCH --mem=64G

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
TASK="lexical"
TASK_PERIOD="production"
RECOMPUTE="false"

while getopts s:b:t:p:r: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        p) TASK_PERIOD=${OPTARG};;
        r) RECOMPUTE=${OPTARG};;
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
echo "Generating spectrogram channel map for subject ${SUBJECT} with task ${TASK} and task period ${TASK_PERIOD}"
# python preprocessing/extract_ieeg_epochs.py \
python generate_specChanMap.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    task=${TASK} \
    task_periods=${TASK_PERIOD} \
    recompute=${RECOMPUTE} \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}