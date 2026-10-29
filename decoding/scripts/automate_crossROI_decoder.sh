#!/bin/bash

join () {
  local IFS="$1"
  shift
  echo "$*"
}

# subj_arr=("S81")
# subj_arr=("S41" "S45" "S47" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78")
# bids_root="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
# task="lexical"
# phonemeIdx=-1


tar_subj_arr=("D0019" "D0022" "D0023" "D0024" "D0025" "D0028" "D0029" "D0035"
              "D0040" "D0041" "D0042" "D0045" "D0049" "D0052" "D0053" "D0054" "D0055"
              "D0056" "D0057" "D0058" "D0059" "D0060" "D0061" "D0063" "D0064" "D0066"
              "D0067" "D0068" "D0069" "D0070" "D0071" "D0073" "D0075" "D0077"
              "D0079" "D0084" "D0085" "D0086" "D0088" "D0091" "D0092" "D0093"
              "D0095" "D0096" "D0100" "D0102" "D0103")
src_subj_arr=("D0019" "D0022" "D0023" "D0024" "D0025" "D0028" "D0029" "D0035"
              "D0040" "D0041" "D0042" "D0045" "D0049" "D0052" "D0053" "D0054" "D0055"
              "D0056" "D0057" "D0058" "D0059" "D0060" "D0061" "D0063" "D0064" "D0066"
              "D0067" "D0068" "D0069" "D0070" "D0071" "D0073" "D0075" "D0077"
              "D0079" "D0084" "D0085" "D0086" "D0088" "D0091" "D0092" "D0093"
              "D0095" "D0096" "D0100" "D0102" "D0103")
# subj_arr=("D0028" "D0040" "D0042" "D0049" "D0052" "D0055" "D0064" "D0068" "D0069" "D0075" "D0079" "D0084" "D0092" "D0095" "D0096")
# subj_arr=("D0028" "D0040" "D0042" "D0049" "D0052" "D0055" "D0064" "D0068" "D0069" "D0075" "D0079" "D0084" "D0092" "D0096")
# subj_arr=("D0095")
task="PhonemeSequence"
# bids_root="/hpc/home/zms14/cworkspace/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(CAR)"
bids_root="/hpc/home/zms14/cworkspace/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
# results_dir="/hpc/home/zms14/cworkspace/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/decoding"
# phonemeIdx=(-1 1 2 3)
# phonemeIdx=(1 2 3)
phonemeIdx=(-1)
n_phons=3

target_rois=("SMC")
source_rois=("Auditory")

nFolds=20
nIter=10
twMin=-1
twMax=1

# task_periods=("production" "perception")
task_periods=("production")
# norm_type="MeanSub"
norm_type="Zscore"
suffix="highGamma"
# suffix="spikeBand"
extension=".fif"

output_suffix=""

datatype="epoch(band)(power)(sig)"
# datatype="epoch(band)(power)"

tar_subj_str=$(join , "${tar_subj_arr[@]}")
src_subj_str=$(join , "${src_subj_arr[@]}")
src_roi_str=$(join , "${source_rois[@]}")

for roi in "${target_rois[@]}"; do
    for task_period in "${task_periods[@]}"; do
        description="${task_period}${norm_type}"
        for phon in "${phonemeIdx[@]}"; do
            echo "----------------------------------------"
            echo "Creating cross-ROI decoding job from ROIS ${src_roi_str} to ROI ${roi}, task_period ${task_period}, phonemeIdx ${phon}"   
            job_name="${roi}_${task_period}_${phon}_${poolTask}_decode_crossROI"
            echo $job_name
            sbatch -J $job_name run_crossROI_decoder.sh \
                -r ${roi} -c ${src_roi_str} -s ${tar_subj_str} -j ${src_subj_str}\
                -b "${bids_root}" -t ${task} \
                -p ${phon} -f ${nFolds} -i ${nIter} \
                -w ${twMin} -x ${twMax} -n ${n_phons} \
                -d ${description} -u ${suffix} -e ${extension} \
                -a "${datatype}"
            echo "----------------------------------------"
        done
    done
done
