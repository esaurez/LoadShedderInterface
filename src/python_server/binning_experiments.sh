
#for sv in "${svlist[@]}"
#for sr in "${srlist[@]}"; do
#   for i in "${!svlist[@]}"; do 
#      sv=${svlist[$i]}
#      . generate_hsb_config.sh $sv $i
#      python3 request_handler.py configs/hsb_config_${i}.properties # trains and stores model
#      python3 test_shedding.py -p configs/hsb_config_${i}.properties -d $sr -r ../test_results/test_hsb_${sr}_mc_fixed_ci/${i} #needs to write to output file.
#   done
#  python3 ../plot_results/compute_false_positives_and_negatives.py -r ../test_results/test_hsb_${sr}_mc_fixed_ci -sv 6 
#done

# another loop for each query here. 

query='red'
for mbs in  0.05 0.1 0.2; do # 0.15 0.2; do
    for mode in 'all_colors' 'max_cdf'; do
        # generate title
        title="${query}_${mbs}_${mode}"
        . generate_hsb_config.sh $mode $title $mbs
        python3 request_handler.py configs/hsb_config_$title.properties # trains and stores model
        for ratio in 0.2 0.4 0.6; do
                python3 test_shedding.py -p configs/hsb_config_$title.properties  -d $ratio -r ../test_results/$query/$ratio/$mbs/$mode
        done
    done
done


# for the plot, I need to loop over the arrays again or something. 
python3 ../plot_results/compute_false_positives_and_negatives.py -r ../test_results/$query -sr 0.2 0.4 0.6 -mbs  0.05 0.1 0.2 -m 'all_colors' 'max_cdf'