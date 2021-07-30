#!/bin/sh
for i in 128 64 32 16
do
  for j in 100 1000 10000
  do
    . generate_config.sh $i $j
  done
done

for i in 128 64 32 16
do
  for j in 100 1000 10000
  do
    python request_handler.py config_${i}_${j}.properties
    echo "trained ${i} ${j}"
    for k in 0.1 0.3 0.5 0.7 0.9
    do
      python test_shedding.py -p config_${i}_${j}.properties -d $k -r ../test_${i}_${j}_${k}
      echo "tested ${i} ${j} ${k}"
      python ../plot_results/compute_false_positives_and_negatives.py -r ../test_${i}_${j}_${k} -t ../final_csv.csv -bs ${i} -fbs ${j} -rt ${k}
      echo "plotted ${i} ${j} ${k}"
    done
  done
done
