"""
Check how supervise pretraining effect the language drift in lewis
"""
from drift.lewis.pretrain import train
from drift.lewis.gumbel import selfplay
from itertools import product
from pandas import DataFrame

train_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

statss = []
for train_size in train_sizes:
    print(train_size)
    speaker, listener, pretrain_stats = train(train_batch_size=5,
                                              train_size=train_size)
    selfplay_stats = selfplay(speaker=speaker, listener=listener)
    stats = {'sl/{}'.format(key): val for key, val in pretrain_stats.items()}
    stats.update({'sp/{}'.format(key): val for key, val in selfplay_stats.items()})
    stats.update({'train_size': train_size})
    statss.append(stats)

# generate CSV
columns = list(statss[0].keys())
data = [[stats[key] for key in stats] for stats in statss]
df = DataFrame(data, columns=columns)
df.to_csv('lewis_exp.csv')
print(df)
