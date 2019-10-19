"""
Baseline selfplay
"""
import torch
import os
from shutil import rmtree
from tensorboardX import SummaryWriter
from drift.lewis import USE_GPU
from drift.lewis.core import LewisGame, get_comm_acc, eval_loop, Dataset
from drift.lewis.gumbel import selfplay_batch
from drift.lewis.linear import Speaker, Listener
import argparse

TRAIN_STEPS = 10000
LOG_STEPS = 10
LOG_NAME = 'log_gumbel'


def selfplay(speaker, listener, gumbel_temperature=0.1):
    """ Train speaker and listener with gumbel softmax. Return stats. """
    if USE_GPU:
        speaker = speaker.cuda()
        listener = listener.cuda()
    game = LewisGame(**speaker.env_config)
    dset = Dataset(game, 1)

    s_opt = torch.optim.Adam(lr=5e-5, params=speaker.parameters())
    l_opt = torch.optim.Adam(lr=5e-5, params=listener.parameters())
    if os.path.exists(LOG_NAME):
        rmtree(LOG_NAME)
    writer = SummaryWriter(LOG_NAME)

    for step in range(TRAIN_STEPS):
        if step % LOG_STEPS == 0:
            stats, s_conf_mat, l_conf_mat = eval_loop(dset.val_generator(1000), listener=listener,
                                                      speaker=speaker, game=game)
            writer.add_image('s_conf_mat', s_conf_mat.unsqueeze(0), step)
            writer.add_image('l_conf_mat', l_conf_mat.unsqueeze(0), step)
            stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
            logstr = ["step {}:".format(step)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
                writer.add_scalar(name, val, step)
            print(' '.join(logstr))
            if stats['comm_acc'] > 0.98:
                stats['step'] = step
                break

        # Train for a batch
        selfplay_batch(game, gumbel_temperature, l_opt, listener, s_opt, speaker)

    stats, s_conf_mat, l_conf_mat = eval_loop(dset.val_generator(1000), listener=listener,
                                              speaker=speaker, game=game)
    stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
    stats['step'] = TRAIN_STEPS
    return stats, speaker, listener


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-speaker', required=True, help='path to speaker pth')
    parser.add_argument('-listener', required=True, help='path to listener pth')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    speaker = Speaker.load(args.speaker)
    listener = Listener.load(args.listener)
    stats, speaker, listener = selfplay(speaker, listener)
    logstr = []
    for name, val in stats.items():
        logstr.append("{}: {:.4f}".format(name, val))
    print(' '.join(logstr))
