"""
Try using population
    1. Training against many listener and many speakers
"""
import os
import argparse
import random
import torch
from tensorboardX import SummaryWriter
from drift.lewis.core import LewisGame, Dataset, eval_loop, get_comm_acc
from drift.lewis.pretrain import train_listener_until, train_speaker_until
from drift.lewis.linear import Listener, Speaker
from drift.lewis.gumbel import selfplay_batch


STEPS = 10000
LOG_STEPS = 10


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-prepare', action='store_true', help='prepare population')
    parser.add_argument('-ckpt_dir', required=True, help='path to save/load ckpts')
    parser.add_argument('-n', default=3, help="population size")
    return parser.parse_args()


def prepare_population(args):
    """ Prepare models and save in ckpt_dir """
    # Prepare n pairs of speakers and listeners with 0.2
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    for i in range(args.n):
        speaker, _ = train_speaker_until(0.2)
        listener, _ = train_listener_until(0.2)
        speaker.save(os.path.join(args.ckpt_dir, 's{}.pth'.format(i)))
        listener.save(os.path.join(args.ckpt_dir, 'l{}.pth'.format(i)))


def _load_population_and_opts(args, s_ckpts, l_ckpts):
    s_and_opts = []
    l_and_opts = []
    for i in range(args.n):
        speaker = Speaker.load(os.path.join(args.ckpt_dir, s_ckpts[i]))
        s_opt = torch.optim.Adam(lr=5e-5, params=speaker.parameters())
        s_and_opts.append([speaker, s_opt])

        listener = Listener.load(os.path.join(args.ckpt_dir, l_ckpts[i]))
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
    writer = SummaryWriter('log_population')

    # Training
    for step in range(STEPS):
        # Randomly pick one pair
        speaker, s_opt = random.choice(s_and_opts)
        listener, l_opt = random.choice(l_and_opts)

        # Train for a Batch
        selfplay_batch(game, 0.1, l_opt, listener, s_opt, speaker)

        # Eval and Logging
        if step % LOG_STEPS == 0:
            stats, _, _ = eval_loop(dset.val_generator(1000), listener=listener,
                                    speaker=speaker, game=game)
            #writer.add_image('s_conf_mat', s_conf_mat.unsqueeze(0), step)
            #writer.add_image('l_conf_mat', l_conf_mat.unsqueeze(0), step)
            stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
            logstr = ["step {}:".format(step)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
                writer.add_scalar(name, val, step)
            print(' '.join(logstr))
            if stats['comm_acc'] > 0.98:
                stats['step'] = step
                break


if __name__ == '__main__':
    args = get_args()
    if args.prepare:
        prepare_population(args)
    else:
        population_selfplay(args)
