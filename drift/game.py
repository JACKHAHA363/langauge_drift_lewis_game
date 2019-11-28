""" Game and dataset """
import argparse
import torch
from drift import USE_GPU, EVALUATION_RATIO
from drift.utils import combine_generator
from itertools import product
import numpy as np


class LewisGame:
    fields = [('p', int, 6, 'nb properties'),
              ('t', int, 10, 'nb types'),
              ('su_ratio', float, 0.05, 'ratio of objects being labeled for supervise'),
              ('sp_ratio', float, 0.1, 'ratio of objects using selfplay. Must be >= su_ratio ')]

    @staticmethod
    def get_parser(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        for name, dtype, val, desc in LewisGame.fields:
            parser.add_argument('-' + name, default=val, type=dtype, help=desc)
        return parser

    @classmethod
    def get_default_config(cls):
        return {name: default for name, _, default, _ in cls.fields}

    def __init__(self, p, t, su_ratio, sp_ratio, **kwargs):
        assert sp_ratio >= su_ratio
        assert 0 < sp_ratio < 1 and 0 < su_ratio < 1
        print('Building a game with p: {}, t: {}'.format(p, t))
        self.p = p
        self.t = t
        self.all_objs = torch.LongTensor([obj for obj in product(*[[t for t in range(self.t)] for _ in range(self.p)])])

        # The offset vector of converting to msg
        self.msg_offset = np.arange(0, self.p) * self.t
        self.msg_offset = torch.Tensor(self.msg_offset).long()
        self.all_msgs = self.objs_to_msg(self.all_objs)

        # Move to GPU
        if USE_GPU:
            self.all_objs = self.all_objs.cuda()
            self.all_msgs = self.all_msgs.cuda()
            self.msg_offset = self.msg_offset.cuda()

        # Build the dataset
        total_size = self.all_objs.shape[0]
        su_size = int(su_ratio * total_size)
        sp_size = int(sp_ratio * total_size)
        print('We have total {} examples, {} for supervise and {} for selfplay'.format(total_size,
                                                                                       su_size,
                                                                                       sp_size))
        self.all_indices = [i for i in range(self.all_objs.shape[0])]
        np.random.shuffle(self.all_indices)
        self.sp_indices = self.all_indices[:sp_size].copy()
        self.su_indices = self.all_indices[:su_size].copy()
        self.heldout_indices = self.all_indices[sp_size:].copy()

    def random_sp_objs(self, batch_size):
        indices = torch.randint(len(self.sp_indices), size=[batch_size]).long()
        batch_ids = self.sp_indices[indices]
        return self.all_objs[batch_ids]

    @property
    def vocab_size(self):
        return self.t * self.p

    def objs_to_msg(self, objs):
        """ Generate the ground truth language for objects
        :param [bsz, nb_props]
        :return [bsz, nb_props]
        """
        return objs + self.msg_offset.unsqueeze(0)

    @property
    def env_config(self):
        return {'p': self.p, 't': self.t}

    def get_generator(self, batch_size, names):
        """
        :param batch_size: Batch size
        :param names: A list of name of generator. From 'su', 'sp', 'heldout'
        """
        names = set(names)
        gen_list = []
        for name in names:
            if name == 'su':
                gen_list.append(self._su_generator(batch_size))
            elif name == 'sp':
                gen_list.append(self._sp_generator(batch_size))
            elif name == 'heldout':
                gen_list.append(self._heldout_generator(batch_size))
            else:
                raise ValueError('Incorrect generator name {}'.format(names))
        return combine_generator(gen_list)

    def _su_generator(self, batch_size):
        np.random.shuffle(self.su_indices)
        return self.create_generator(self.all_objs[self.su_indices],
                                     self.all_msgs[self.su_indices],
                                     batch_size)

    def _sp_generator(self, batch_size):
        np.random.shuffle(self.sp_indices)
        return self.create_generator(self.all_objs[self.sp_indices],
                                     self.all_msgs[self.sp_indices],
                                     batch_size)

    def _heldout_generator(self, batch_size):
        """ Used for evaluation. For each validation loop, randomly pick EVALUATION_RATIO * heldout_objects
        """
        split = int(EVALUATION_RATIO * len(self.heldout_indices))
        np.random.shuffle(self.heldout_indices)
        final_ids = self.heldout_indices[:split]

        # Take the validation data
        objs = self.all_objs[final_ids]
        msgs = self.all_msgs[final_ids]
        return self.create_generator(objs, msgs, batch_size)

    @staticmethod
    def create_generator(objs, msgs, batch_size):
        start = 0
        while start < objs.shape[0]:
            batch_objs, batch_msgs = objs[start: start + batch_size], msgs[start: start + batch_size]
            if USE_GPU:
                batch_objs = batch_objs.cuda()
                batch_msgs = batch_msgs.cuda()
            yield batch_objs, batch_msgs
            start += batch_size


if __name__ == '__main__':
    game = LewisGame(p=1, t=10, su_ratio=0.2, sp_ratio=0.5)
    for obj, _ in game.su_generator(1):
        print('su', obj)
    for obj, _ in game.su_generator(1):
        print('su', obj)

    for obj, _ in game.sp_generator(1):
        print('sp', obj)
