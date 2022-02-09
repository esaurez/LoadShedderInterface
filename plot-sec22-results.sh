#!/bin/bash

source ~/venv/bin/activate
main_dir=$(realpath ./sec22-plots)
rm -r $main_dir
mkdir $main_dir

# Plotting the red only results
mkdir $main_dir/red

cd ~/LoadShedderInterface/src/color_analysis/sv_matrix/new_feat_sv_mat

./cross_validation.sh ../../training_confs/10fs/red_cheat/ $main_dir/red/

python3 ../plot_cross_validation_results.py -U $main_dir/red/frame_utils.csv -O $main_dir/red/ -C ../../training_confs/10fs/red_cheat/

python3 ../random_shedding.py -C ../../training_confs/10fs/red_cheat/ -U $main_dir/red/frame_utils.csv -O $main_dir/red/ -R 0.5
cd -

# Plotting Red OR Yellow results
D=$main_dir/OR
mkdir $D

cd ~/LoadShedderInterface/src/color_analysis/sv_matrix/new_feat_sv_mat/composite
./cross_validation.sh ../../../training_confs/10fs/red_cheat/ ../../../training_confs/10fs/yellow_cheat/ OR $D
python3 ../../composite/plot_cross_validation_results.py -U $D/frame_utils.csv -O $D --composite-or -C ../../../training_confs/10fs/red_cheat/ ../../../training_confs/10fs/yellow_cheat/
cd -

# Plotting Red AND Yellow results
D=$main_dir/AND
mkdir $D

cd ~/LoadShedderInterface/src/color_analysis/sv_matrix/new_feat_sv_mat/composite
./cross_validation.sh ../../../training_confs/10fs/red_cheat/ ../../../training_confs/10fs/yellow_cheat/ AND $D
python3 ../../composite/plot_cross_validation_results.py -U $D/frame_utils.csv -O $D -C ../../../training_confs/10fs/red_cheat/ ../../../training_confs/10fs/yellow_cheat/
cd -

# Plotting E2E results
D=$main_dir/E2E
mkdir $D
cd ~/LoadShedderInterface/src/e2e_sim
python3 plot_qor_for_e2e.py -C ../color_analysis/training_confs/10fs/red_cheat/ -I ./shedding_decisions/ -D 500 --fps 10 -O $D
cd -

# Plotting latency breakdown
D=$main_dir/runtime_breakdown
mkdir $D
cd ~/LoadShedderInterface/src/runtime_analysis
python3 plot_runtime_breakdown.py -D data.log -O $D
cd -
