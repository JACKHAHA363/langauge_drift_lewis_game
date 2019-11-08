"""
This code will prepare the population with desired accuracy
Usage: python prepare_population.py -ckpt_dir zzz_divese_pop -s_arch linear -l_arch linear -n 3 -acc 0.2
"""
import os
from drift.core import LewisGame, Dataset
from drift.pretrain import train_speaker_until, train_listener_until
from drift.arch import get_listener_cls, get_speaker_cls
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt_dir', required=True, help='path to save/load ckpts')
    parser.add_argument('-s_arch', required=True, help='speaker arch')
    parser.add_argument('-l_arch', required=True, help='listener arch')
    parser.add_argument('-n', type=int, default=3, help="population size")
    parser.add_argument('-sacc', type=float, default=0.2, help='speak supervise training acc')
    parser.add_argument('-lacc', type=float, default=0.2, help='listen supervise training acc')
    parser.add_argument('-dset_size', type=int, default=10000, help='size of training dataset')
    parser.add_argument('-switch_dset', action='store_true', help='If true each model use different dataset')
    parser.add_argument('-partial_dset', action='store_true', help='Learn from the same dataset each time '
                                                                   'with partial data')
    parser = LewisGame.get_parser(parser)
    return parser.parse_args()


def prepare_population(args):
    s_cls = get_speaker_cls(args.s_arch)
    l_cls = get_listener_cls(args.l_arch)
    env_config = {'p': args.p, 't': args.t}
    game = LewisGame(**env_config)
    dset = Dataset(train_size=args.dset_size, game=game)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    for i in range(args.n):
        if args.switch_dset:
            dset = Dataset(train_size=args.dset_size, game=game)
        elif args.partial_dset:
            dset.use_partial_dataset()
        speaker, _ = train_speaker_until(args.sacc, s_cls(env_config), dset)
        if args.switch_dset:
            dset = Dataset(train_size=args.dset_size, game=game)
        elif args.partial_dset:
            dset.use_partial_dataset()
        listener, _ = train_listener_until(args.lacc, l_cls(env_config), dset)
        speaker.save(os.path.join(args.ckpt_dir, 's{}.pth'.format(i)))
        listener.save(os.path.join(args.ckpt_dir, 'l{}.pth'.format(i)))


if __name__ == '__main__':
    args = get_args()
    if args.switch_dset:
        print('Train on different dataset')
        if args.partial_dset:
            print('switch_dset and partial_dset cannot both be True')
            exit()
    else:
        print('Train on same dataset')
        if args.partial_dset:
            print('Each train on different parts of same dataset')
    prepare_population(args)
