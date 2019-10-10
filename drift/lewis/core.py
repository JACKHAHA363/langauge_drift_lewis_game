"""
Lewis Signal Game
"""
import torch
import argparse
from itertools import product


class LewisGame:
    @classmethod
    def get_cmd_parser(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--p', type=int, default=6, help='nb properties')
        parser.add_argument('--t', type=int, default=10, help='nb types')
        return parser

    def __init__(self, p, t):
        self.p = p
        self.t = t
        self.all_objs = torch.LongTensor([obj for obj in product(*[[t for t in range(self.t)] for _ in range(self.p)])])

    def get_random_objs(self, batch_size):
        return torch.randint(low=0, high=self.t, size=[batch_size, self.p])

    @property
    def vocab_size(self):
        return self.t * self.p

    def objs_to_msg(self, objs):
        """ Generate the ground truth language for objects
        :param [bsz, nb_props]
        :return [bsz, nb_props]
        """
        addition = torch.Tensor([i for i in range(self.p)])
        return objs + addition.unsqueeze(0) * self.t


class Speaker:
    """ Speaker model """
    def output_msg(self, objs):
        """
        :param objs: [bsz, nb_props*nb_types]
        :return: msgs: [bsz, nb_props]
        """
        pass

    def get_msg_logprobs(self, objs, msg):
        pass


class Listener:
    """ Listener """
    def build_obj(self, msgs):
        """
        :param msgs: [bsz, nb_props]
        :return: objs: [bsz, nb_props]
        """

    def get_obj_logprobs(self, msgs, objs):
        pass

