import torch
from torch.distributions import Categorical
from drift.core import LewisGame, Dataset, eval_listener_loop, eval_speaker_loop
import numpy as np

VAL_BATCH_SIZE = 1000
LOG_STEPS = 10
MAX_STEPS = 2000


def listener_imitate(student_listener, l_opt, teacher_listener, max_steps):
    game = LewisGame(**student_listener.env_config)
    dset = Dataset(game=game, train_size=1)
    step = 0
    accs = []
    try:
        while True:
            if step >= max_steps:
                break

            # Generate batch with teacher listener
            msgs = game.objs_to_msg(game.get_random_objs(50))
            oh_msgs = student_listener.one_hot(msgs)
            with torch.no_grad():
                teacher_logits = teacher_listener(oh_msgs)
                #objs = torch.distributions.Categorical(logits=teacher_logits).sample()
                objs = torch.argmax(teacher_logits, -1)

            # Train this batch
            train_listener_batch(student_listener, l_opt, objs, msgs)
            step += 1

            # Evaluate
            if step % LOG_STEPS == 0:
                l_corrects = 0
                l_total = 0
                for _, msgs in dset.val_generator(1000):
                    with torch.no_grad():
                        oh_msgs = student_listener.one_hot(msgs)
                        teacher_logits = teacher_listener(oh_msgs)
                        #objs = torch.distributions.Categorical(logits=teacher_logits).sample()
                        objs = torch.argmax(teacher_logits, -1)
                        l_logits = student_listener(oh_msgs)
                        l_pred = torch.argmax(l_logits, dim=-1)
                        l_corrects += (l_pred == objs).float().sum().item()
                        l_total += objs.numel()
                stats = {'l_acc': l_corrects / l_total}
                accs.append(stats['l_acc'])

                # Report
                logstr = ["step {}:".format(step)]
                for name, val in stats.items():
                    logstr.append("{}: {:.4f}".format(name, val))
                print(' '.join(logstr))
                if stats['l_acc'] >= 0.95:
                    break
    except KeyboardInterrupt:
        pass
    return accs


def speaker_imitate(student_speaker, s_opt, teacher_speaker, max_steps):
    game = LewisGame(**student_speaker.env_config)
    dset = Dataset(game=game, train_size=1)
    step = 0
    accs = []
    try:
        while True:
            if step >= max_steps:
                break

            # Generate batch with teacher listener
            objs = game.get_random_objs(50)
            with torch.no_grad():
                teacher_logits = teacher_speaker(objs)
                #msgs = torch.distributions.Categorical(logits=teacher_logits).sample()
                msgs = torch.argmax(teacher_logits, -1)

            # Train this batch
            train_speaker_batch(student_speaker, s_opt, objs, msgs)
            step += 1

            # Evaluation
            if step % LOG_STEPS == 0:
                s_corrects = 0
                s_total = 0
                for objs, _ in dset.val_generator(1000):
                    with torch.no_grad():
                        teacher_logits = teacher_speaker(objs)
                        #msgs = torch.distributions.Categorical(logits=teacher_logits).sample()
                        msgs = torch.argmax(teacher_logits, -1)
                        s_logits = student_speaker(objs)
                        s_pred = torch.argmax(s_logits, dim=-1)
                        s_corrects += (s_pred == msgs).float().sum().item()
                        s_total += msgs.numel()
                stats = {'s_acc': s_corrects / s_total}

                # Report
                logstr = ["step {}:".format(step)]
                for name, val in stats.items():
                    logstr.append("{}: {:.4f}".format(name, val))
                print(' '.join(logstr))
                accs.append(stats['s_acc'])
                if stats['s_acc'] >= 0.95:
                    break
    except KeyboardInterrupt:
        pass
    return accs


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
    try:
        while True:
            should_stop = (step >= MAX_STEPS) or should_stop
            if should_stop:
                break

            for objs, msgs in dset.train_generator(5):
                if step >= MAX_STEPS:
                    should_stop = True
                    break
                train_speaker_batch(speaker, s_opt, objs, msgs)
                step += 1
                if step % LOG_STEPS == 0:
                    stats = eval_speaker_loop(dset.val_generator(VAL_BATCH_SIZE),
                                              speaker=speaker)
                    logstr = ["step {}:".format(step)]
                    for name, val in stats.items():
                        logstr.append("{}: {:.4f}".format(name, val))
                        print(' '.join(logstr))
                    if stats['s_acc'] >= acc:
                        should_stop = True
                        break
    except KeyboardInterrupt:
        pass
    return speaker, stats


def train_listener_until(acc, listener, dset):
    """ Train listener until desired acc """
    l_opt = torch.optim.Adam(lr=5e-4, params=listener.parameters())

    should_stop = False
    step = 0
    stats = None
    try:
        while True:
            should_stop = (step >= MAX_STEPS) or should_stop
            if should_stop:
                break

            for objs, msgs in dset.train_generator(5):
                if step >= MAX_STEPS:
                    should_stop = True
                    break
                train_listener_batch(listener, l_opt, objs, msgs)
                step += 1
                stats = eval_listener_loop(dset.val_generator(VAL_BATCH_SIZE),
                                           listener=listener)
                logstr = ["step {}:".format(step)]
                for name, val in stats.items():
                    logstr.append("{}: {:.4f}".format(name, val))
                print(' '.join(logstr))
                if stats['l_acc'] >= acc:
                    should_stop = True
                    break
    except KeyboardInterrupt:
        pass

    return listener, stats
