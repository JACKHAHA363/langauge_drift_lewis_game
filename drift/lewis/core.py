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
    def __init__(self):
        super(Agent, self).__init__()

    @abstractmethod
    def save(self, pth_path):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, pth_path):
        raise NotImplementedError


class BaseSpeaker(Agent):
    """ Speaker model """
    @abstractmethod
    def output_msg(self, objs):
        """ Method for testing
        :param objs: [bsz, nb_props*nb_types]
        :return: msgs: [bsz, nb_props]
        """
        raise NotImplementedError

    @abstractmethod
    def get_msg_logprobs(self, objs, msg):
        """ logprobs [bsz, nb_props] """
        raise NotImplementedError


class BaseListener(Agent):
    """ Listener """
    @abstractmethod
    def build_obj(self, msgs):
        """ Method for testing
        :param msgs: [bsz, nb_props]
        :return: objs: [bsz, nb_props]
        """
        raise NotImplementedError

    @abstractmethod
    def get_obj_logprobs(self, msgs, objs):
        """ logprobs [bsz, nb_props] """
        raise NotImplementedError
