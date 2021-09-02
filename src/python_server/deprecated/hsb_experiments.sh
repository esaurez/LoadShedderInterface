#!/bin/sh

# property file generieren

splitvalues=500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000  # its simply a string. though I need to iterate over it.
splitvalues=1000,10000,100000,
splitvalues=1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,30000,40000,50000,60000
splitvalues=5,10,15,20,25,30,35
splitvalues=10,20,30,40,50,60,70,80
splitvalues=30,
splitvalues=1,2,3,4,5,6,7,8,9,10,15,20,25,30
splitvalues=20
splitvalues=5,10,20,30,

title="test_hsb_5_to_30" # iterate over that one, too. or maybe just make it manually...



. generate_hsb_config.sh $splitvalues $title

echo configs/hsb_config_${title}.properties

python3 request_handler.py configs/hsb_config_${title}.properties # trains and stores model

    echo "trained hsb model"
    for k in 0.01 0.05 0.1 0.3 0.5 0.7 0.9
    do
      python3 test_shedding.py -p configs/hsb_config_$title.properties -d $k -r ../test_results/test_hsb_${title}/${k} #ok
    done

    python3 ../plot_results/compute_false_positives_and_negatives.py -r ../test_results/test_hsb_$title -t ../final_csv.csv -sr 0.01 0.05 0.1 0.3 0.5 0.7 0.9

mv configs/hsb_config_${title}.properties ../test_results/test_hsb_$title/hsb_config_${title}.properties

