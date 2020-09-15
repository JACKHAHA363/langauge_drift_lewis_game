#!/bin/bash
K1=2000
K2=200
ROOT_EXP_DIR='./exps'
CKPT_DIR="${ROOT_EXP_DIR}/zzz_ckpt"
GUMBEL_TEMP=10
EXP_DIR="${ROOT_EXP_DIR}/iter_generation${K1}_transmission${K2}_same_opt"
LOG_DIR=${EXP_DIR}

python iterated_learning.py -method gumbel -ckpt_dir ${CKPT_DIR} -generation_steps ${K1} \
          -s_transmission_steps ${K2} -l_transmission_steps ${K2} -logdir ${LOG_DIR} \
        	-temperature ${GUMBEL_TEMP} -decay_rate 1. -s_lr 0.0001 -l_lr 0.0001 -steps 90000 -batch_size 50 \
        	-log_steps 500 -same_opt