import torch
from torch.distributions import Categorical
from drift.core import eval_listener_loop, eval_speaker_loop
import numpy as np

VAL_BATCH_SIZE = 1000
LOG_STEPS = 10
MAX_STEPS = 5000


def imitate_listener_batch(student, teacher, opt, msgs, temperature=0):
    """ Imitate teacher on this batch. If temperature > 0, it's imitate soft label.
        Else imitate argmax
    """
    # Generate target obj
    with torch.no_grad():
        oh_msgs = teacher.one_hot(msgs)
        teacher_logits = teacher.get_logits(oh_msgs)

    # Train with argmax
    if temperature == 0:
        objs = torch.argmax(teacher_logits, -1)
        train_listener_batch(student, opt, objs, msgs)

    else:
        soft_label = torch.nn.functional.softmax(teacher_logits / temperature, -1)
        student_logits = student.get_logits(oh_msgs)
        student_logprobs = torch.nn.functional.log_softmax(student_logits, -1)
        loss = -(soft_label * student_logprobs).sum(-1).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def imitate_speak_batch(student, teacher, opt, objs, temperature=0, use_sample=False):
    """ Imitate teacher on this batch. If temperature > 0, it's imitate soft label.
        Else imitate argmax
    """
    # Generate context msg
    with torch.no_grad():
        msgs = teacher.greedy(objs)

    # Train with argmax
    if temperature == 0:
        train_speaker_batch(student, opt, objs, msgs)

    elif not use_sample:
        teacher_logits = teacher.get_logits(msgs=msgs, objs=objs)
        soft_label = torch.nn.functional.softmax(teacher_logits / temperature, -1)
        student_logits = student.get_logits(objs=objs, msgs=msgs)
        student_logprobs = torch.nn.functional.log_softmax(student_logits, -1)
        loss = -(soft_label * student_logprobs).sum(-1).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    else:
        teacher_logits = teacher.get_logits(msgs=msgs, objs=objs)
        msgs = torch.distributions.Categorical(logits=teacher_logits / temperature).sample()
        train_speaker_batch(student, opt, objs, msgs)


def train_listener_batch(listener, l_opt, objs, msgs):
    l_logits = listener.get_logits(listener.one_hot(msgs))
    l_logprobs = Categorical(logits=l_logits).log_prob(objs)
    l_opt.zero_grad()
    (-l_logprobs.mean()).backward()
    l_opt.step()


def train_speaker_batch(speaker, s_opt, objs, msgs):
    """ Perform a train step """
    s_logits = speaker.get_logits(objs, msgs)
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
                    if stats['speak/tf_acc'] >= acc:
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
