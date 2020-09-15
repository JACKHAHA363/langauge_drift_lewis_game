ROOT_EXP_DIR='./exps'
CKPT_DIR="${ROOT_EXP_DIR}/zzz_ckpt"

python prepare_population.py -ckpt_dir ${CKPT_DIR} \
  -s_arch linear -l_arch linear -n 1 -p 5 -t 5 -sacc 1 -lacc 1
