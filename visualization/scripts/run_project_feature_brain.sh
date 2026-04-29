#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/brain_projection/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/brain_projection/%j.error
#SBATCH -p common,scavenger
#SBATCH -c 10
#SBATCH --mem=32G

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
SUBJECTS_DIR="/hpc/home/zms14/cworkspace/ECoG_Recon"
FEATURES="high_gamma"
TASK_PERIOD="[perception,production]"

while getopts s:b:t:d:f:p: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        d) SUBJECTS_DIR=${OPTARG};;
        f) FEATURES=${OPTARG};;
        p) TASK_PERIOD=${OPTARG};;
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
echo "Projecting features onto brain for subject ${SUBJECT}"
python project_feature_brain.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    task=${TASK} \
    "subjects_dir='${SUBJECTS_DIR}'" \
    features=${FEATURES} \
    task_periods=${TASK_PERIOD} \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
