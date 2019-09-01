#!/usr/bin/env bash

############################### 
# Training a mt-dnn model
# Note that this is a toy setting and please refer our paper for detailed hyper-parameters.
############################### 

srun -w vanadium -c 14 --mem 10000 -J mt-dnn python prepro.py
srun -w vanadium -c 14 --mem 10000 -J mt-dnn python train.py
