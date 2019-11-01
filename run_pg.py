"""
Baseline selfplay with PG
"""
import torch
import os
from shutil import rmtree
from tensorboardX import SummaryWriter
from drift.core import LewisGame, get_comm_acc, eval_loop, Dataset
# from drift.pg import selfplay_batch, ExponentialMovingAverager
from drift.a2c import selfplay_batch, ExponentialMovingAverager
from drift.linear import Speaker, Listener
import argparse
import numpy as np

TRAIN_STEPS = 10000
LOG_STEPS = 10


def selfplay(args, speaker, listener):
    game = LewisGame(**speaker.env_config)
    dset = Dataset(game, 1)

    s_opt = torch.optim.Adam(lr=5e-5, params=speaker.parameters())
    l_opt = torch.optim.Adam(lr=5e-5, params=listener.parameters())
    if os.path.exists(args.log):
        rmtree(args.log)
    writer = SummaryWriter(args.log)

    ema_reward = None
    for step in range(TRAIN_STEPS):
        if step % LOG_STEPS == 0:
            stats, s_conf_mat, l_conf_mat = eval_loop(dset.val_generator(1000), listener=listener,
                                                      speaker=speaker, game=game)
            writer.add_image('s_conf_mat', s_conf_mat.unsqueeze(0), step)
            writer.add_image('l_conf_mat', l_conf_mat.unsqueeze(0), step)
            stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
            logstr = ["epoch {}:".format(step)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
                writer.add_scalar(name, val, step)
            print(' '.join(logstr))
            if ema_reward:
                writer.add_histogram('Value Function', ema_reward.mean, step)
                writer.add_histogram('Value Function 2', np.concatenate((ema_reward.mean,-ema_reward.mean)), step)

                writer.add_histogram('value nums', ema_reward.num, step)

            if stats['comm_acc'] > 0.98:
                stats['step'] = step
                break

        # Train a batch
        ema_reward = selfplay_batch(game, l_opt, listener, s_opt, speaker, ema_reward)

    stats, s_conf_mat, l_conf_mat = eval_loop(dset.val_generator(1000), listener=listener,
                                              speaker=speaker, game=game)
    stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
    stats['step'] = TRAIN_STEPS
    return stats, speaker, listener


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-speaker', required=True, help='path to speaker pth')
    parser.add_argument('-listener', required=True, help='path to listener pth')
    parser.add_argument('-log', required=True, help='Name of log')
    # parser.add_argument('-a2c', action='store_false', default=False, dest='a2c', help='Use A2C')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    speaker = torch.load(args.speaker)
    listener = torch.load(args.listener)
    stats, speaker, listener = selfplay(args, speaker, listener)
    logstr = []
    for name, val in stats.items():
        logstr.append("{}: {:.4f}".format(name, val))
    print(' '.join(logstr))
