"""
Lewis Signal Game
"""
import torch
import argparse
from itertools import product
from abc import abstractmethod


class LewisGame:
    fields = [('p', int, 6, 'nb properties'),
              ('t', int, 10, 'nb types')]

    @classmethod
    def get_cmd_parser(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        for name, dtype, default, desc in cls.fields:
            parser.add('--' + name, type=dtype, default=default, help=desc)
        return parser

    @classmethod
    def get_default_config(cls):
        return {name: default for name, _, default, _ in cls.fields}

    def __init__(self, p, t):
        self.p = p
        self.t = t
        self.all_objs = torch.LongTensor([obj for obj in product(*[[t for t in range(self.t)] for _ in range(self.p)])])

    def get_random_objs(self, batch_size):
        return torch.randint(low=0, high=self.t, size=[batch_size, self.p]).long()

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
        speaker = cls(env_config=['env_config'])
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
