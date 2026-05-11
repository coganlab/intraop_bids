#!/bin/bash
#SBATCH --output=/hpc/home/zms14/cworkspace/jobs/spatial_avg_raw/%j.out
#SBATCH -e /hpc/home/zms14/cworkspace/jobs/spatial_avg_raw/%j.error
#SBATCH -p common,scavenger
#SBATCH -c 8
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
TASK="lexical"
BIDS_ROOT="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS"
CONTACT_SIZES="[2,4,8]"

while getopts s:b:t:z: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        z) CONTACT_SIZES=${OPTARG};;
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
echo "Generating spatial-average raw derivatives for subject ${SUBJECT} (contact_sizes=${CONTACT_SIZES})"
python spatial_avg_raw.py \
    patient=${SUBJECT} \
    "bids_root='${BIDS_ROOT}'" \
    task=${TASK} \
    "contact_sizes=${CONTACT_SIZES}" \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
