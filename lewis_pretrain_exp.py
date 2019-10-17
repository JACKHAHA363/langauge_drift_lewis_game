"""
Check how supervise pretraining effect the language drift in lewis
"""
from drift.lewis.pretrain import train
from drift.lewis.gumbel import selfplay
from itertools import product
from pandas import DataFrame

batch_sizes = [5, 25, 45]
train_sizes = [50, 100, 150]
train_stepss = [20, 50, 80, 100]

statss = []
for batch_size, train_size, train_step in product(batch_sizes, train_sizes, train_stepss):
    print(batch_size, train_size, train_step)
    speaker, listener, pretrain_stats = train(train_batch_size=batch_size, train_steps=train_step,
                                              train_size=train_size)
    selfplay_stats = selfplay(speaker=speaker, listener=listener)
    stats = {'sl/{}'.format(key): val for key, val in pretrain_stats.items()}
    stats.update({'sp/{}'.format(key): val for key, val in selfplay_stats.items()})
    stats.update({'train_batch_size': batch_size,
                  'train_size': train_size,
                  'train_step': train_step})
    statss.append(stats)

# generate CSV
columns = list(statss[0].keys())
data = [[stats[key] for key in stats] for stats in statss]
df = DataFrame(data, columns=columns)
df.to_csv('lewis_exp.csv')
print(df)
