#!/bin/sh
#sv1=5,
#sv2=10,
#sv3=20,
#sv4=30,
#sv5=40,
#sv6=50,

sv1=5,10,15,20,25  #5er Schritte
sv2=5,15,25,35 #10er Schritte start bei 5
sv3=10,20,30,40 # 10er Schritte Start bei 10
sv4=5,10,20,30 # 10er Schritt und 1 kleiner
sv5=5,25,45 #20er Schritt Start bei 5
sv6=10,40,60 #20er Schritt Start bei 10


#sv5=5,10,20,30,40,
#sv6=5,10,15,20,25,
#sv7=5,10,20,30,40,50,60,
#sv8=10,15,20,25,30,35,

sr=0.5


svlist=($sv1 $sv2 $sv3 $sv4 $sv5 $sv6)

#for sv in "${svlist[@]}"
for i in "${!svlist[@]}"; do 
   sv=${svlist[$i]}
   . generate_hsb_config.sh $sv $i
   python3 request_handler.py configs/hsb_config_${i}.properties # trains and stores model
   python3 test_shedding.py -p configs/hsb_config_${i}.properties -d $sr -r ../test_results/test_hsb_multiple_classes/${i} #needs to write to output file.
done

python3 ../plot_results/compute_false_positives_and_negatives.py -r ../test_results/test_hsb_multiple_classes -sv 6 # anzahl experimente übergeben dann aus config holen? 
# für's plotten brauch ich ja gar keine split values.