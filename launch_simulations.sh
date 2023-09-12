#!/bin/sh

conditions=( "normal" "halved" "quarter" "tenth" "hundreth")

for i in "${conditions[@]}" ; do
    code=$(qsub "genn-network-model/cluster/"$i"_synapses_poisson.pbs" | awk -F '.' '{print $1}')
    qsub -W depend=afterok:$code "genn-network-model/cluster/"$i"_synapses_poisson_extract_feature.pbs"
    qsub -W depend=afterok:$code "genn-network-model/cluster/"$i"_synapses_poisson_sdf.pbs"
    qsub -W depend=afterok:$code "genn-network-model/cluster/"$i"_synapses_poisson_correlation.pbs"
done
