#! /bin/bash

dataset=geonames
seed='1357'
bhole=500
verbose=True
batch=1
result_dir=result/$dataset

criteria='jaccard'
trigger='random'
lb_strategy='by_criteria'
holes='10.1720'
restore='T'
rs="True"
contamination=0.05
test_config=test_config.adult
splitratio='11,fold'
npivots='10'
nbits='10'
methods="maxfreq"
portions="0 1 2 3 4 5 6 7 8 9 10"

for hole in $holes; do
    for s in $seed; do
       for method in $methods; do
           for npivot in $npivots; do
               for nbit in $nbits; do
                   for portion in $portions; do
                        G="$(cut -d'.' -f1 <<<"$hole")"
                        H="$(cut -d'.' -f2 <<<"$hole")"
echo `date`: start experiment for dataset $dataset, criteria $criteria, lb_strategy $lb, trigger $t, seed $s, G $G, H $H, nbit, $nbit, npivot, $npivot, pivotmethod, $method, portion, $portion
                                
python3 main.py --result-dir $result_dir --dataset $dataset --seed $s --num-holes $G --data-perhole $H --trigger $trigger --verbose $verbose --batch $batch --lb-strategy $lb_strategy --criteria $criteria --test-config $test_config --restore $restore  --bhole $bhole --contamination $contamination --split-ratio $splitratio --nbit $nbit --npivot $npivot --pivot-method $method --split-portion $portion --test-nn Y 

echo `date`: end experiment for dataset $dataset, criteria $criteria, lb_strategy $lb, trigger $t, seed $s, G $G, H $H, nbit, $nbit, npivot, $npivot, pivotmethod, $method, portion, $portion
                    done
                done
			done
		done
	done
done
