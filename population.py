"""
Try using population
    1. Training against many listener and many speakers
"""
import os
import argparse
import random
import torch
from shutil import rmtree
from tensorboardX import SummaryWriter
from drift.core import LewisGame, Dataset, eval_loop, get_comm_acc
from drift.gumbel import selfplay_batch
from drift import USE_GPU

STEPS = 40000
LOG_STEPS = 100


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt_dir', required=True, help='path to save/load ckpts')
    parser.add_argument('-logdir', required=True, help='path to tb log')
    parser.add_argument('-n', type=int, default=3, help="population size")
    return parser.parse_args()


def _load_population_and_opts(args, s_ckpts, l_ckpts):
    s_and_opts = []
    l_and_opts = []
    for i in range(args.n):
        speaker = torch.load(os.path.join(args.ckpt_dir, s_ckpts[i]))
        if USE_GPU:
            speaker = speaker.cuda()
        s_opt = torch.optim.Adam(lr=5e-5, params=speaker.parameters())
        s_and_opts.append([speaker, s_opt])

        listener = torch.load(os.path.join(args.ckpt_dir, l_ckpts[i]))
        if USE_GPU:
            listener = listener.cuda()
        l_opt = torch.optim.Adam(lr=5e-5, params=listener.parameters())
        l_and_opts.append([listener, l_opt])
    return s_and_opts, l_and_opts


def population_selfplay(args):
    """ Load checkpoints """
    # Make sure it's valid ckpt dirs
    all_ckpts = os.listdir(args.ckpt_dir)
    s_ckpts = ['s{}.pth'.format(i) for i in range(args.n)]
    for s_ckpt in s_ckpts:
        assert s_ckpt in all_ckpts
    l_ckpts = ['l{}.pth'.format(i) for i in range(args.n)]
    for l_ckpt in l_ckpts:
        assert l_ckpt in all_ckpts

    # Load populations
    s_and_opts, l_and_opts = _load_population_and_opts(args, s_ckpts, l_ckpts)
    game = LewisGame(**s_and_opts[0][0].env_config)
    dset = Dataset(game, 1)
    if os.path.exists(args.logdir):
        rmtree(args.logdir)
    writer = SummaryWriter(args.logdir)

    # Training
    for step in range(STEPS):
        # Randomly pick one pair
        speaker, s_opt = random.choice(s_and_opts)
        listener, l_opt = random.choice(l_and_opts)

        # Train for a Batch
        selfplay_batch(game, 1, l_opt, listener, s_opt, speaker)

        # Eval and Logging
        if step % LOG_STEPS == 0:
            stats, _, _ = eval_loop(dset.val_generator(1000), listener=listener,
                                    speaker=speaker, game=game)
            stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
            logstr = ["step {}:".format(step)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
                writer.add_scalar(name, val, step)
            writer.flush()
            print(' '.join(logstr))
            #if stats['comm_acc'] == 1.:
            #    stats['step'] = step
            #    break


if __name__ == '__main__':
    args = get_args()
    population_selfplay(args)
