import torch
from torch.distributions import Categorical
from drift.core import LewisGame, Dataset, eval_listener_loop, eval_speaker_loop
import numpy as np

VAL_BATCH_SIZE = 1000
LOG_STEPS = 10


def train_listener_batch(listener, l_opt, objs, msgs):
    l_logits = listener(listener.one_hot(msgs))
    l_logprobs = Categorical(logits=l_logits).log_prob(objs)
    l_opt.zero_grad()
    (-l_logprobs.mean()).backward()
    l_opt.step()


def train_speaker_batch(speaker, s_opt, objs, msgs):
    """ Perform a train step """
    s_logits = speaker(objs)
    s_logprobs = Categorical(logits=s_logits).log_prob(msgs)
    s_opt.zero_grad()
    (-s_logprobs.mean()).backward()
    s_opt.step()


class EarlyStopper:
    def __init__(self, eps=2e-3, patience=3):
        self.val = 0
        self.time = 0
        self.eps = eps
        self.patience = patience

    def should_stop(self, val):
        if np.abs((val - self.val)) < self.eps:
            self.time += 1
        else:
            self.time = 0
        self.val = val
        return self.time == self.patience


def train_speaker_until(acc, speaker, dset):
    """ Return a speaker trained until desired acc. If speaker is None construct a default one.
    """
    s_opt = torch.optim.Adam(lr=1e-4, params=speaker.parameters())

    should_stop = False
    step = 0
    stats = None
    while True:
        if should_stop:
            break

        for objs, msgs in dset.train_generator(5):
            train_speaker_batch(speaker, s_opt, objs, msgs)
            step += 1
            if step % LOG_STEPS == 0:
                stats = eval_speaker_loop(dset.val_generator(VAL_BATCH_SIZE),
                                          speaker=speaker)
                logstr = ["step {}:".format(step)]
                for name, val in stats.items():
                    logstr.append("{}: {:.4f}".format(name, val))
                    print(' '.join(logstr))
                if stats['s_acc'] > acc:
                    should_stop = True
                    break

    return speaker, stats


def train_listener_until(acc, listener, dset):
    """ Train listener until desired acc """
    l_opt = torch.optim.Adam(lr=5e-4, params=listener.parameters())

    should_stop = False
    step = 0
    stats = None
    while True:
        if should_stop:
            break

        for objs, msgs in dset.train_generator(5):
            train_listener_batch(listener, l_opt, objs, msgs)
            step += 1
            stats = eval_listener_loop(dset.val_generator(VAL_BATCH_SIZE),
                                       listener=listener)
            logstr = ["step {}:".format(step)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
            print(' '.join(logstr))
            if stats['l_acc'] > acc:
                should_stop = True
                break

    return listener, stats
