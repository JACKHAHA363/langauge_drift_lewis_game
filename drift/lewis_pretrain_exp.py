"""
Check how supervise pretraining effect the language drift in lewis
"""
from drift.lewis.pretrain import train
from drift.lewis.gumbel import selfplay
from itertools import product

batch_sizes = [10, 20, 30, 40, 50]
train_sizes = [20, 40, 60, 80, 100]
train_stepss = [40, 80, 120, 160, 200]

for batch_size, train_size, train_step in product(batch_sizes, train_sizes, train_stepss):
    print(batch_size, train_size, train_step)
    speaker, listener, pretrain_stats = train(train_batch_size=batch_size, train_steps=train_step,
                                              train_size=train_size)
    selfplay_stats = selfplay()

