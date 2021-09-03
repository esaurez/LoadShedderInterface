# iterates over: 
# mbs: minimum bin size (required share of all samples per bin)
# mode: all_colors or max_cdf
# ratio: shedding ratio

query='red' # set the correct path to the respective testing and training data in the hsb_configs_template!!! 

for mbs in  0.05 0.1 0.2; do # 0.15 0.2; do
    for mode in 'all_colors' 'max_cdf'; do
        # generate title
        title="${query}_${mbs}_${mode}"
        ./generate_hsb_config.sh $mode $title $mbs
        python3 request_handler.py configs/hsb_config_$title.properties # trains and stores model
        for ratio in 0.2 0.4 0.6; do
                python3 test_shedding.py -p configs/hsb_config_$title.properties  -d $ratio -r ../test_results/$query/$ratio/$mbs/$mode
        done
    done
done


# -sr shedding ratios used in the loop
# -mbs minimum bin sizes 
# -m modes
python3 ../plot_results/compute_false_positives_and_negatives.py -r ../test_results/$query -sr 0.2 0.4 0.6 -mbs  0.05 0.1 0.2 -m 'all_colors' 'max_cdf'
