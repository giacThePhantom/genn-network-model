#!/bin/sh


simulation=("t30noinputcluster" "t30noinputpoissoncluster" "t30noinputhalvedsynapsescluster" "t30noinputhalvedsynapsespoissoncluster" "t30noinputquartersynapsescluster" "t30noinputquartersynapsespoissoncluster" "t30noinputtenthsynapsescluster" "t30noinputtenthsynapsespoissoncluster")

for i in "${simulation[@]}"; do
    for j in "${simulation[@]}"; do
        if [ "$i" != "$j" ]; then
            echo "Comparing $i and $j"
            python -m beegenn.plots.comparing_correlations /media/data/thesis_output/cluster/outputs $i $j pearson
            python -m beegenn.plots.comparing_correlations /media/data/thesis_output/cluster/outputs $i $j spearman
        fi
    done
done
