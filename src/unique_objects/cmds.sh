source ~/venv/bin/activate

for i in seed10484-1-train  seed335500-0-train  seed949563-1-train  seed949563-3-train seed10484-3-train  seed949563-0-train  seed949563-2-train
do

    curr_bin=$(ls ~/LoadShedderInterface/data/10fps/red_only/$i/*.bin)
    echo $curr_bin
    python3 gen_fixed_labels_csv.py -I ./extended_dump/red_10fps/$i/new.txt -O ./extended_dump/red_10fps/$i/

    python3 ../color_analysis/fix_bin_file_labels.py -L ./extended_dump/red_10fps/$i/updated_frames_labels.csv -B $curr_bin -O ~/LoadShedderInterface/data/10fps/red_only/$i
    
    python3 gen_unique_objs_per_frame.py -I ./extended_dump/red_10fps/$i/new.txt -O ~/LoadShedderInterface/data/10fps/red_only/$i

    mv ~/LoadShedderInterface/data/10fps/red_only//$i/updated_labels.bin $curr_bin
done

