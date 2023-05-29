#!/bin/sh

python -m beegenn.simulation t_36 t36 &
python -m beegenn.simulation t_36 t36noinput &

wait

python -m beegenn.simulation t_30 t30 &
python -m beegenn.simulation t_30 t30noinput &

wait


python -m beegenn.plots.sdf t_36 t36 &
python -m beegenn.plots.correlation t_36 t36 &

wait

python -m beegenn.plots.sdf t_36 t36noinput &
python -m beegenn.plots.correlation t_36 t36noinput &

wait

python -m beegenn.plots.sdf t_30 t30 &
python -m beegenn.plots.correlation t_30 t30 &

wait

python -m beegenn.plots.sdf t_30 t30noinput &
python -m beegenn.plots.correlation t_30 t30noinput &
  # | tee >(xargs -I {} python -m beegenn.simulation data {}) \
  #   >(python -m beegenn.plot.varying_odor_param)
