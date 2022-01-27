# Generate the updated_frames_labels.csv file from the object lifetime file manually (created called new.txt)
#python3 gen_fixed_labels_csv.py -I ./extended_dump/red_only/seed10484-1-train/new.txt -O ./extended_dump/red_only/seed10484-1-train/

# Fix the labels in the bin file using the updated_frames_labels.csv file
#for i in seed10484-1-train seed10484-2-train seed10484-3-train; do python3 ../color_analysis/fix_bin_file_labels.py -L ./extended_dump/red_only/$i/updated_frames_labels.csv -B ~/LoadShedderInterface/data/red_fixed_labels/$i/*.bin -O ~/LoadShedderInterface/data/red_fixed_labels/$i; done

# Generate the objs file in the format that the codde understand`
#for i in seed10484-1-train seed10484-2-train seed10484-3-train; do python3 gen_unique_objs_per_frame.py -I ./extended_dump/red_only/$i/new.txt -O ~/LoadShedderInterface/data/red_fixed_labels/$i; done

source ~/venv/bin/activate

for i in seed335500-0-train seed335500-1-train seed335500-2-train seed335500-3-train #seed949563-0-train seed949563-1-train seed949563-2-train seed949563-3-train
do

    curr_bin=$(ls ~/LoadShedderInterface/data/red_fixed_labels/$i/*.bin)
    echo $curr_bin
    python3 gen_fixed_labels_csv.py -I ./extended_dump/red_only/$i/new.txt -O ./extended_dump/red_only/$i/

    python3 ../color_analysis/fix_bin_file_labels.py -L ./extended_dump/red_only/$i/updated_frames_labels.csv -B $curr_bin -O ~/LoadShedderInterface/data/red_fixed_labels/$i
    
    python3 gen_unique_objs_per_frame.py -I ./extended_dump/red_only/$i/new.txt -O ~/LoadShedderInterface/data/red_fixed_labels/$i

    mv ~/LoadShedderInterface/data/red_fixed_labels/$i/updated_labels.bin $curr_bin
done

