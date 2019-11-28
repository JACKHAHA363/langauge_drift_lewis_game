"""
This code will prepare the population with desired accuracy
Usage: python prepare_population.py -ckpt_dir zzz_divese_pop -s_arch linear -l_arch linear -n 3 -acc 0.2
The resulting ckpt will have
s0.pth, l0.pth,
s1.pth, l0.pth,
...
sn.pth, ln.pth
and finally
game.pth is save iff. switch dset is False
"""
import os
import torch
from drift.game import LewisGame
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
    parser.add_argument('-switch_dset', action='store_true', help='If true each model use different dataset')
    parser = LewisGame.get_parser(parser)
    return parser.parse_args()


def prepare_population(args):
    s_cls = get_speaker_cls(args.s_arch)
    l_cls = get_listener_cls(args.l_arch)
    game = LewisGame(**args.__dict__)
    if not args.switch_dset:
        with open(os.path.join(args.ckpt_dir, 'game.pth'), 'wb') as f:
            torch.save(game, f)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    for i in range(args.n):
        # Change dataset for each pair
        if args.switch_dset:
            game = LewisGame(**args.__dict__)
        speaker, _ = train_speaker_until(args.sacc, s_cls(game.env_config), game)
        listener, _ = train_listener_until(args.lacc, l_cls(game.env_config), game)
        speaker.save(os.path.join(args.ckpt_dir, 's{}.pth'.format(i)))
        listener.save(os.path.join(args.ckpt_dir, 'l{}.pth'.format(i)))


if __name__ == '__main__':
    args = get_args()
    if args.switch_dset:
        print('Train on different dataset')
    else:
        print('Train on same dataset')
    prepare_population(args)
