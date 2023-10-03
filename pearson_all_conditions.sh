#!/bin/sh

sim=("t30noinputcluster" "t30noinputpoissoncluster" "t30noinputhalvedsynapsespoissoncluster" "t30noinputquartersynapsespoissoncluster" "t30noinputtenthsynapsespoissoncluster" "t30noinputhundrethsynapsespoissoncluster")


length=${#sim[@]}

for ((i=0; i<$length; i++))
do
    for ((j=(($i+1)); j<$length; j++))
    do
        echo "Running ${sim[$i]} and ${sim[$j]}"
        python -m beegenn.plots.comparing_correlations /media/data/thesis_output ${sim[$i]} ${sim[$j]} pearson
    done
done
