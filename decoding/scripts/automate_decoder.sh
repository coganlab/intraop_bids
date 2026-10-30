#!/bin/bash

# subj_arr=("S86")
# # subj_arr=("S30" "S41" "S45" "S47" "S51" "S52" "S53" "S55" "S56" "S63" "S67" "S73" "S74" "S75" "S76" "S78" "S81" "S82" "S83")
# bids_root="~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
# phonemeIdx=(-1 1 2 3 4 5)
# # phonemeIdx=(1 2 3 4 5)
# # phonemeIdx=(-1)
# # phonemeIdx=(-1 1)
# nPhons=5
# task="lexical"

# # subj_arr=("S14")
# subj_arr=("S14" "S16" "S18" "S19" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")
# # subj_arr=("S16" "S18" "S19" "S22" "S23" "S26" "S32" "S33" "S36" "S39" "S57" "S58" "S62")
# bids_root="~/cworkspace/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
# phonemeIdx=(-1 1 2 3)
# # phonemeIdx=(1 2 3)
# # phonemeIdx=(-1)
# nPhons=3
# task="phoneme"

# subj_arr=("D0095")
# subj_arr=("D0019" "D0022" "D0023" "D0024" "D0025" "D0028" "D0029" "D0031" "D0035"
#           "D0040" "D0041" "D0042" "D0045" "D0049" "D0052" "D0053" "D0054" "D0055"
#           "D0056" "D0057" "D0058" "D0059" "D0060" "D0061" "D0063" "D0064" "D0066"
#           "D0067" "D0068" "D0069" "D0070" "D0071" "D0073" "D0075" "D0076" "D0077"
#           "D0079" "D0084" "D0085" "D0086" "D0088" "D0091" "D0092" "D0093" "D0094"
#           "D0095" "D0096" "D0100" "D0102" "D0103")
subj_arr=("D0023" "D0024" "D0025" "D0028" "D0029" "D0031" "D0035" "D0040" "D0041"
          "D0042" "D0045" "D0049" "D0052" "D0053" "D0054" "D0055" "D0056" "D0057"
          "D0058" "D0059" "D0060" "D0061" "D0063" "D0064" "D0066" "D0067" "D0068"
          "D0069" "D0070" "D0071" "D0073" "D0075" "D0077" "D0079" "D0084" "D0085"
          "D0086" "D0088" "D0091" "D0092" "D0093" "D0094" "D0095" "D0096" "D0100"
          "D0102" "D0103")
# bids_root="/hpc/home/zms14/cworkspace/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(CAR)"
bids_root="/hpc/home/zms14/cworkspace/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(phonemeLevel)(CAR)"
# bids_root="/hpc/home/zms14/cworkspace/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(phonemeLevel)(bipolar)"
# phonemeIdx=(-1 1 2 3)
# phonemeIdx=(1)
phonemeIdx=(-1)
nPhons=3
task="PhonemeSequence"

nFolds=10
nIter=50
# chance=true
chance=false
twMin=-0.5
twMax=0.5

# datatype="epoch(band)(power)(sig)"
datatype="epoch(band)(power)"

# task_periods=("perception" "production")
task_periods=("production")
# norm_type="MeanSub"
norm_type="Zscore"
suffix="highGamma"
# suffix="spikeBand"
# suffix='[highGamma,spikeBand]'
# suffix='lowGamma'
# suffix='beta'
# suffix='alpha'
# suffix='theta'
# suffix='delta'
# suffix='[highGamma,lowGamma]'
# suffix='[highGamma,beta]'
# suffix='[highGamma,alpha]'
# suffix='[highGamma,theta]'
# suffix='[highGamma,delta]'
extension=".fif"


for subject in "${subj_arr[@]}"; do
    for task_period in "${task_periods[@]}"; do
        description="${task_period}${norm_type}"
        for phon in "${phonemeIdx[@]}"; do
            echo "----------------------------------------"
            echo "Creating processing job for ${subject}, task_period ${task_period}, phonemeIdx ${phon}"
            job_name="${subject}_${task_period}_${phon}_decode_phonemeLevel"
            echo $job_name
            sbatch -J $job_name run_decoder.sh \
                -s ${subject} -b "${bids_root}" -t ${task} \
                -p ${phon} -n ${nPhons} -f ${nFolds} -i ${nIter} \
                -w ${twMin} -x ${twMax} \
                -d ${description} -u ${suffix} -e ${extension} \
                -a "${datatype}" -c ${chance}
            echo "----------------------------------------"
        done
    done
done
