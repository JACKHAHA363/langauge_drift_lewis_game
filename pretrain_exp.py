"""
Check how supervise pretraining effect the language drift in lewis
"""
from drift.pretrain import train_speaker_until, train_listener_until
from run_gumbel import selfplay
from drift.linear import Speaker, Listener
from itertools import product
from pandas import DataFrame
import argparse

S_ACCS = [0.2, 0.4, 0.6, 0.8]
L_ACCS = [0.2, 0.4, 0.6, 0.8]


def prepare_ckpt():
    """ Generate checkpoints """
    for l_acc in L_ACCS:
        listener, stats = train_listener_until(l_acc)
        print('final acc', stats['l_acc'])
        listener.save('l_{}.pth'.format(l_acc))

    for s_acc in S_ACCS:
        speaker, stats = train_speaker_until(s_acc)
        print('final acc', stats['s_acc'])
        speaker.save('s_{}.pth'.format(s_acc))


def compute_drift_stats(args):
    """ Compute self play stats """
    # Compute self play stats
    statss = []
    for s_acc, l_acc in product(S_ACCS, L_ACCS):
        speaker = Speaker.load('s_{}.pth'.format(s_acc))
        listener = Listener.load('l_{}.pth'.format(l_acc))
        selfplay_stats, _, _ = selfplay(speaker=speaker, listener=listener)
        stats = {'sp/{}'.format(key): val for key, val in selfplay_stats.items()}
        stats.update({'sl/s_acc': s_acc,
                      'sl/l_acc': l_acc})
        statss.append(stats)

    # generate CSV
    columns = list(statss[0].keys())
    data = [[stats[key] for key in stats] for stats in statss]
    df = DataFrame(data, columns=columns)
    df.to_csv(args.csv)
    print(df)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-prepare', action='store_true', help='prepare pretrain checkpoints')
    parser.add_argument('-csv', default='exp.csv', help='output csv name')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.prepare:
        prepare_ckpt()
    else:
        compute_drift_stats(args)
