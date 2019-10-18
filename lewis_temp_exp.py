"""
Check how temperature the language drift in lewis
"""
from drift.lewis.pretrain import train
from drift.lewis.gumbel import selfplay
from itertools import product
from pandas import DataFrame

temperatures = [0.1, 0.2, 0.3, 0.4, 0.5]

statss = []
for temperature in temperatures:
    speaker, listener, _ = train(train_batch_size=5, train_steps=50, train_size=100)
    selfplay_stats, _, _ = selfplay(speaker=speaker, listener=listener, gumbel_temperature=temperature)
    stats = {}
    stats.update({'sp/{}'.format(key): val for key, val in selfplay_stats.items()})
    stats.update({'gumbel_temperature': temperature})
    statss.append(stats)

# generate CSV
columns = list(statss[0].keys())
data = [[stats[key] for key in stats] for stats in statss]
df = DataFrame(data, columns=columns)
df.to_csv('lewis_temp_exp.csv')
print(df)
