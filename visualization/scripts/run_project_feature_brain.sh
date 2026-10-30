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

# Optional overrides for the refactored plotting/projection options.
# Leave empty to fall back to the values in config/project_HG_brain.yaml.
PROJECTION=""        # average | subject
AVERAGE=""           # template brain (e.g. cvs_avg35_inMNI152)
SURFACE=""           # pial | white
HEMI=""              # lh | rh | both | split
RM_WM=""             # true | false (subject projection only)
LABEL_ELECTRODES=""  # true | false (required for rm_wm filtering)

while getopts s:b:t:d:f:p:j:a:e:m:w:l: flag
do
    case "${flag}" in
        s) SUBJECT=${OPTARG};;
        b) BIDS_ROOT=${OPTARG};;
        t) TASK=${OPTARG};;
        d) SUBJECTS_DIR=${OPTARG};;
        f) FEATURES=${OPTARG};;
        p) TASK_PERIOD=${OPTARG};;
        j) PROJECTION=${OPTARG};;
        a) AVERAGE=${OPTARG};;
        e) SURFACE=${OPTARG};;
        m) HEMI=${OPTARG};;
        w) RM_WM=${OPTARG};;
        l) LABEL_ELECTRODES=${OPTARG};;
    esac
done

if [[ -z "${SUBJECT}" ]]; then
    echo "Please specify patient id with -s"
    exit 1
fi

# Only forward optional overrides that were explicitly set, so unset options
# keep their config defaults.
EXTRA_ARGS=()
[[ -n "${PROJECTION}" ]] && EXTRA_ARGS+=("projection=${PROJECTION}")
[[ -n "${AVERAGE}" ]] && EXTRA_ARGS+=("average=${AVERAGE}")
[[ -n "${SURFACE}" ]] && EXTRA_ARGS+=("surface=${SURFACE}")
[[ -n "${HEMI}" ]] && EXTRA_ARGS+=("hemi=${HEMI}")
[[ -n "${RM_WM}" ]] && EXTRA_ARGS+=("rm_wm=${RM_WM}")
[[ -n "${LABEL_ELECTRODES}" ]] && EXTRA_ARGS+=("label_electrodes=${LABEL_ELECTRODES}")

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
    "${EXTRA_ARGS[@]}" \
    hydra.run.dir=/hpc/home/zms14/cworkspace/outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
