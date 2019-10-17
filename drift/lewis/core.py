"""
Lewis Signal Game
"""
import torch
import argparse
from itertools import product
from drift.lewis import USE_GPU, EVALUATION_RATIO
from abc import abstractmethod
import numpy as np
from tqdm import tqdm


class LewisGame:
    fields = [('p', int, 6, 'nb properties'),
              ('t', int, 10, 'nb types')]

    @classmethod
    def get_default_config(cls):
        return {name: default for name, _, default, _ in cls.fields}

    def __init__(self, p, t):
        self.p = p
        self.t = t
        self.all_objs = torch.LongTensor([obj for obj in product(*[[t for t in range(self.t)] for _ in range(self.p)])])
        self.all_msgs = self.objs_to_msg(self.all_objs)

    def get_random_objs(self, batch_size):
        indices = torch.randint(len(self.all_objs), size=[batch_size]).long()
        return self.all_objs[indices]

    @property
    def vocab_size(self):
        return self.t * self.p

    def objs_to_msg(self, objs):
        """ Generate the ground truth language for objects
        :param [bsz, nb_props]
        :return [bsz, nb_props]
        """
        addition = torch.Tensor([i for i in range(self.p)]).long()
        return objs + addition.unsqueeze(0) * self.t


class Agent(torch.nn.Module):
    def __init__(self, env_config):
        super(Agent, self).__init__()
        self.env_config = env_config

    def save(self, pth_path):
        info = {'env_config': self.env_config,
                'state_dict': self.state_dict()}
        torch.save(info, pth_path)

    @classmethod
    def load(cls, pth_path):
        info = torch.load(pth_path)
        speaker = cls(env_config=info['env_config'])
        speaker.load_state_dict(info['state_dict'])
        return speaker


class BaseSpeaker(Agent):
    """ Speaker model """

    def get_logits(self, objs):
        """ Return [bsz, nb_props, vocab_size] """
        return self.forward(objs)


class BaseListener(Agent):
    """ Listener """

    def get_logits(self, msgs):
        """ Method for testing
        :param msgs: [bsz, nb_props]
        :return: objs: [bsz, nb_props, nb_types]
        """
        return self.forward(msgs)


def get_comm_acc(val_generator, listener, speaker):
    corrects = 0
    total = 0
    for objs, _ in val_generator:
        with torch.no_grad():
            s_logits = speaker(objs)
            msgs = torch.argmax(s_logits, dim=-1)
            l_logits = listener(listener.one_hot(msgs))
            preds = torch.argmax(l_logits, dim=-1)
            corrects += (preds == objs).float().sum().item()
            total += objs.numel()
    return {'comm_acc': corrects / total}


def eval_loop(val_generator, listener, speaker):
    l_corrects = 0
    l_total = 0
    s_corrects = 0
    s_total = 0

    # Add speaker confusion matrix
    vocab_size = listener.env_config['p'] * listener.env_config['t']
    s_conf_mat = torch.zeros([vocab_size, vocab_size])
    for objs, msgs in tqdm(val_generator):
        with torch.no_grad():
            l_logits = listener(listener.one_hot(msgs))
            l_pred = torch.argmax(l_logits, dim=-1)
            l_corrects += (l_pred == objs).float().sum().item()
            l_total += objs.numel()

            s_logits = speaker(objs)
            s_pred = torch.argmax(s_logits, dim=-1)
            s_corrects += (s_pred == msgs).float().sum().item()
            s_total += msgs.numel()

            for m, pred in zip(msgs.view(-1), s_pred.view(-1)):
                s_conf_mat[m, pred] += 1
    s_conf_mat /= torch.sum(s_conf_mat, -1, keepdim=True)
    return {'l_acc': l_corrects / l_total, 's_acc': s_corrects / s_total}, s_conf_mat


class Dataset:
    """ The dataset object """
    def __init__(self, game, train_size):
        assert isinstance(game, LewisGame)
        self.train_objs = game.get_random_objs(train_size)
        self.train_msgs = game.objs_to_msg(self.train_objs)
        self.train_size = train_size
        self.game = game
        self.all_indices = [i for i in range(len(game.all_objs))]

    def train_generator(self, batch_size):
        return self._get_generator(self.train_objs, self.train_msgs, batch_size)

    def val_generator(self, batch_size):
        """ Used for evaluation """
        # Shuffled indices
        np.random.shuffle(self.all_indices)
        split = int(EVALUATION_RATIO * len(self.all_indices))
        objs = self.game.all_objs[self.all_indices[:split]]
        msgs = self.game.all_msgs[self.all_indices[:split]]
        return self._get_generator(objs, msgs, batch_size)

    @staticmethod
    def _get_generator(objs, msgs, batch_size):
        start = 0
        while start < len(objs):
            batch_objs, batch_msgs = objs[start: start + batch_size], msgs[start: start + batch_size]
            if USE_GPU:
                batch_objs = batch_objs.cuda()
                batch_msgs = batch_msgs.cuda()
            yield batch_objs, batch_msgs
            start += batch_size
