#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/feature_videos/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/feature_videos/%j.error
#SBATCH -p common,scavenger
#SBATCH -c 5
#SBATCH --mem=8G

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
TASK_PERIOD="[perception,production]"
FEATURES="[high_gamma,spike_band]"

while getopts s:b:t:p:f: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        p) TASK_PERIOD=${OPTARG};;
        f) FEATURES=${OPTARG};;
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
echo "Generating feature videos for subject ${SUBJECT}"
python generate_featureChanMapVids.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    task=${TASK} \
    task_periods=${TASK_PERIOD} \
    features=${FEATURES} \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
