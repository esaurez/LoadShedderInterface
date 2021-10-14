set -o xtrace

# iterates over: 
# mbs: minimum bin size (required share of all samples per bin)
# mode: all_colors or max_cdf
# ratio: shedding ratio

query='red' # set the correct path to the respective testing and training data in the hsb_configs_template!!! 

for mbs in  0.01 0.05 0.1 0.15 0.2 0.25 0.3 ; do # 0.15 0.2; do
    for mode in 'max_cdf' 'random'; do #'random'; do
        # Title for the Load Shedder model
        title="${query}_${mbs}_${mode}"

        # The generate_hsb_config.sh script generates the config file 
        # Each config file has a different Min Bin Size (MBS) and Mode
        ./generate_hsb_config.sh $mode $title $mbs

        # Trains the Utility Function model and stores it.
        # Stored model contains 
        #   - per-color split values (bins)
        #   - utility of each bin
        #   - mapping from drop rate to utility threshold
        python3 request_handler.py configs/hsb_config_$title.properties

        # Now analyze the performance of load shedder with different ratios
        for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
            # IMPORTANT : This Python script can fail when run for the 1st time, because files are not setup. Rerunning seems to solve that error.
            python3 test_shedding.py -p configs/hsb_config_$title.properties  -d $ratio -r ../test_results/$query/$ratio/$mbs/$mode
        done
    done
done

# -sr shedding ratios used in the loop
# -mbs minimum bin sizes 
# -m modes
python3 ../plot_results/compute_false_positives_and_negatives.py -r ../test_results/$query -sr 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 -mbs  0.01 0.05 0.1 0.15 0.2 0.25 0.3  -m 'random' 'max_cdf'
