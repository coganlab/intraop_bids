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

subj_arr=("S14" "S16" "S18" "S19" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")
# subj_arr=("S16" "S18" "S19" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")
# subj_arr=("S14")
bids_root="~/cworkspace/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
task="phoneme"
phonemeIdx=(-1)
n_phons=3

# subj_arr=("D0019" "D0022" "D0023" "D0024" "D0025" "D0028" "D0029" "D0035"
#           "D0040" "D0041" "D0042" "D0045" "D0049" "D0052" "D0053" "D0054" "D0055"
#           "D0056" "D0057" "D0058" "D0059" "D0060" "D0061" "D0063" "D0064" "D0066"
#           "D0067" "D0068" "D0069" "D0070" "D0071" "D0073" "D0075" "D0077"
#           "D0079" "D0084" "D0085" "D0086" "D0088" "D0091" "D0092" "D0093" "D0094"
#           "D0095" "D0096" "D0100" "D0102" "D0103")
# # subj_arr=("D0028" "D0040" "D0042" "D0049" "D0052" "D0055" "D0064" "D0068" "D0069" "D0075" "D0079" "D0084" "D0092" "D0095" "D0096")
# # subj_arr=("D0028" "D0040" "D0042" "D0049" "D0052" "D0055" "D0064" "D0068" "D0069" "D0075" "D0079" "D0084" "D0092" "D0096")
# # subj_arr=("D0095")
# task="PhonemeSequence"
# # bids_root="/hpc/home/zms14/cworkspace/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(CAR)"
# bids_root="/hpc/home/zms14/cworkspace/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
# # results_dir="/hpc/home/zms14/cworkspace/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/decoding"
# # phonemeIdx=(-1 1 2 3)
# # phonemeIdx=(1 2 3)
# phonemeIdx=(-1)
# n_phons=3

cross_subj_arr=(D0028 D0040 D0042 D0049 D0052 D0055 D0064 D0068 D0069 D0075 D0079 D0084 D0092 D0095 D0096)
# cross_subj_arr=("D0019" "D0022" "D0023" "D0024" "D0025" "D0028" "D0029" "D0035"
#                 "D0040" "D0041" "D0042" "D0045" "D0049" "D0052" "D0053" "D0054" "D0055"
#                 "D0056" "D0057" "D0058" "D0059" "D0060" "D0061" "D0063" "D0064" "D0066"
#                 "D0067" "D0068" "D0069" "D0070" "D0071" "D0073" "D0075" "D0077"
#                 "D0079" "D0084" "D0085" "D0086" "D0088" "D0091" "D0092" "D0093"
#                 "D0095" "D0096" "D0100" "D0102" "D0103")
cross_subj_str=$(join , "${cross_subj_arr[@]}")

nFolds=20
nIter=10
twMin=-1
twMax=1

# task_periods=("production" "perception")
task_periods=("production")
norm_type="MeanSub"
# norm_type="Zscore"
suffix="highGamma"
# suffix="spikeBand"
extension=".fif"

datatype="epoch(band)(power)(sig)"

output_suffix=""

base_mem=64
mem_per_subj=16
if [[ $datatype == *"sig"* ]]; then
    mem_per_subj=12
fi

mem_amt=$(( $base_mem + $mem_per_subj * ${#cross_subj_arr[@]} ))
mem="${mem_amt}G"


for subject in "${subj_arr[@]}"; do
    for task_period in "${task_periods[@]}"; do
        description="${task_period}${norm_type}"
        for phon in "${phonemeIdx[@]}"; do
            echo "----------------------------------------"
            echo "Creating processing job for ${subject}, task_period ${task_period}, phonemeIdx ${phon}"
            job_name="${subject}_${task_period}_${phon}_decode_crossPt"
            echo $job_name
            sbatch -J $job_name --mem=$mem run_crossPt_decoder.sh \
                -s ${subject} -b "${bids_root}" -t ${task} \
                -p ${phon} -c ${cross_subj_str} -f ${nFolds} -i ${nIter} \
                -w ${twMin} -x ${twMax} -n ${n_phons} \
                -d ${description} -u ${suffix} -e ${extension} \
                -a "${datatype}" -h ${output_suffix}
            echo "----------------------------------------"
        done
    done
done
