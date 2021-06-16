#!/bin/bash

for dataset in ICEWS14
do
    for method in conv lstm mlp selfatt
    do
        for run in 1 2 3 
        do
            rm ${dataset}_${method}_${run}.out
            python train.py --data ${dataset} --config config/${dataset}/${method}.yml --gpu 2 > ${dataset}_${method}_${run}.out
        done
    done
done