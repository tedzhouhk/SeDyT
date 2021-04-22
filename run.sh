#!/bin/bash

for step in $(seq 12 1 20)
do
    python train.py --data WIKI --config config/wiki_conv.yml --force_step ${step} > step_out/${step}.out
done