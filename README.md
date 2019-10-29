# A set of toy environment
1. OR (Object Reconstruction) Game :heavy_check_mark:
2. Image Referential Game

[Reference](https://openreview.net/pdf?id=rJxGLlBtwH)

# Install
`git clone` and `pip intall -e .`

# Usage
prepare populations:
```
python prepare_population.py -ckpt_dir zzz_diverse_pop -s_arch linear -l_arch linear -n 3 -acc 0.2
```
Run single pair of agent:
```
python population.py -ckpt_dir zzz_diverse_pop -n 1
```
Run simple population
```
python population.py -ckpt_dir zzz_diverse_pop -n 3
```

# List of Arch
1. linear
2. dropout
