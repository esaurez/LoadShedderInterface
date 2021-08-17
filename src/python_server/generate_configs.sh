#!/bin/sh

#TODO sowohl bin size als auch scaling factor sind ja jetzt nicht mehr die zu Nutzenden Werte!!!
for i in 128 64 32 16
do
  for j in 100 #1000 10000
  do
    . generate_config.sh $i $j
  done
done

for i in 128 64 32 16
do
  for j in 100 #1000 10000
  do
    python3 request_handler.py configs/config_${i}_${j}.properties # trains and stores model
    echo "trained ${i} ${j}"
    for k in 0.1 0.3 0.5 0.7 0.9
    do
      python3 test_shedding.py -p configs/config_${i}_${j}.properties -d $k -r ../test_results/test_${i}_${j}_${k}
      echo "tested ${i} ${j} ${k}"
      python3 ../plot_results/compute_false_positives_and_negatives.py -r ../test_results/test_${i}_${j}_${k} -t ../final_csv.csv -bs ${i} -fbs ${j} -rt ${k}
      echo "plotted ${i} ${j} ${k}"
    done
  done
done
