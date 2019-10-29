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
    parser.add_argument('-acc', type=float, default=0.2, help='supervise training acc')
    return parser.parse_args()


def prepare_population(args):
    s_cls = get_speaker_cls(args.s_arch)
    l_cls = get_listener_cls(args.l_arch)
    env_config = LewisGame.get_default_config()
    game = LewisGame(**env_config)
    dset = Dataset(train_size=100000, game=game)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    for i in range(args.n):
        speaker, _ = train_speaker_until(args.acc, s_cls(env_config), dset)
        listener, _ = train_listener_until(args.acc, l_cls(env_config), dset)
        speaker.save(os.path.join(args.ckpt_dir, 's{}.pth'.format(i)))
        listener.save(os.path.join(args.ckpt_dir, 'l{}.pth'.format(i)))


if __name__ == '__main__':
    args = get_args()
    prepare_population(args)
