#!/bin/sh
sim_names=$(python -m beegenn.parameters.generate_parameters \
  data/protocols/experiment1.json \
  data/protocols/explore_different_odors_with_sigma.json \
  data/simulations/cor_no_self_100_od_2_conc.json \
  data/simulations/explore_different_odors_with_sigma.json)

#echo $sim_names | xargs -d ' ' -I {} python -m beegenn.simulation data {}

echo $sim_names | python -m beegenn.plots.varying_odors_param data explore_different_odors_with_sigma




  # | tee >(xargs -I {} python -m beegenn.simulation data {}) \
  #   >(python -m beegenn.plot.varying_odor_param)
