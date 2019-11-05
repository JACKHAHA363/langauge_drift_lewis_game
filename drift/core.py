"""
Lewis Signal Game
"""
import torch
from torch.nn.functional import softmax
from itertools import product
from drift import USE_GPU, EVALUATION_RATIO
from drift.utils import timeit
import numpy as np


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

        # The offset vector of converting to msg
        self.msg_offset = np.arange(0, self.p) * self.t
        #np.random.shuffle(self.msg_offset)
        self.msg_offset = torch.Tensor(self.msg_offset).long()
        self.all_msgs = self.objs_to_msg(self.all_objs)

        # Move to GPU
        if USE_GPU:
            self.all_objs = self.all_objs.cuda()
            self.all_msgs = self.all_msgs.cuda()
            self.msg_offset = self.msg_offset.cuda()

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
        return objs + self.msg_offset.unsqueeze(0)


class Agent(torch.nn.Module):
    def __init__(self, env_config):
        super(Agent, self).__init__()
        self.env_config = env_config

    def save(self, pth_path):
        torch.save(self, pth_path)

    @classmethod
    def from_state_dict(cls, env_config, state_dict):
        agent = cls(env_config)
        agent.load_state_dict(state_dict)
        return agent


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


@timeit('get_comm_acc')
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


def eval_speaker_loop(val_generator, speaker):
    """ Return stats """
    s_corrects = 0
    s_total = 0
    for objs, msgs in val_generator:
        with torch.no_grad():
            s_logits = speaker(objs)
            s_pred = torch.argmax(s_logits, dim=-1)
            s_corrects += (s_pred == msgs).float().sum().item()
            s_total += msgs.numel()
    return {'s_acc': s_corrects / s_total}


def eval_listener_loop(val_generator, listener):
    l_corrects = 0
    l_total = 0
    for objs, msgs in val_generator:
        with torch.no_grad():
            l_logits = listener(listener.one_hot(msgs))
            l_pred = torch.argmax(l_logits, dim=-1)
            l_corrects += (l_pred == objs).float().sum().item()
            l_total += objs.numel()
    return {'l_acc': l_corrects / l_total}


def _obj_prob_to_msg_prob(obj_probs):
    """
    :param obj_probs: [nb_obj, nb_type, nb_value]
    :return: [nb_obj, nb_type * nb_value]
    """
    nb_obj, nb_type, nb_value = obj_probs.shape[0], obj_probs.shape[1], obj_probs.shape[2]
    result = torch.zeros([nb_obj, nb_type, nb_type * nb_value])
    result = result.to(device=obj_probs.device)
    for i in range(nb_type):
        start = i * nb_value
        end = (i+1) * nb_value
        result[:, i, start:end] = obj_probs[:, i]
    return result


@timeit('eval_loop')
def eval_loop(val_generator, listener, speaker, game):
    """ Return accuracy as well as confusion matrix for symbols """
    l_corrects = 0
    l_total = 0
    s_corrects = 0
    s_total = 0

    # Add speaker confusion matrix
    vocab_size = listener.env_config['p'] * listener.env_config['t']
    s_conf_mat = torch.zeros([vocab_size, vocab_size])
    l_conf_mat = torch.zeros([vocab_size, vocab_size])
    for objs, msgs in val_generator:
        with torch.no_grad():
            l_logits = listener(listener.one_hot(msgs))
            l_pred = torch.argmax(l_logits, dim=-1)
            l_probs = softmax(l_logits, dim=-1)
            l_corrects += (l_pred == objs).float().sum().item()
            l_total += objs.numel()

            s_logits = speaker(objs)
            s_pred = torch.argmax(s_logits, dim=-1)
            s_probs = softmax(s_logits, dim=-1)
            s_corrects += (s_pred == msgs).float().sum().item()
            s_total += msgs.numel()

            s_conf_mat[msgs.view(-1)] += s_probs.view([-1, vocab_size])
            l_conf_mat[msgs.view(-1)] += _obj_prob_to_msg_prob(l_probs).view([-1, vocab_size])

    s_conf_mat /= (1e-32 + torch.sum(s_conf_mat, -1, keepdim=True))
    l_conf_mat /= (1e-32 + torch.sum(l_conf_mat, -1, keepdim=True))
    return {'l_acc': l_corrects / l_total, 's_acc': s_corrects / s_total}, s_conf_mat, l_conf_mat


class Dataset:
    """ The dataset object """

    def __init__(self, game, train_size):
        assert isinstance(game, LewisGame)
        self.train_objs = game.get_random_objs(train_size)
        self.train_msgs = game.objs_to_msg(self.train_objs)
        self.game = game

        # Shuffle all objects index
        self.all_indices = [i for i in range(len(game.all_objs))]
        np.random.shuffle(self.all_indices)
        self.train_inds = self.all_indices[:train_size].copy()
        self.actual_train_inds = self.train_inds

        # Reshuffle all indice to get valid objects
        np.random.shuffle(self.all_indices)
        self.valid_start = 0

    def use_partial_dataset(self):
        """ Adjust actual train inds to partial """
        np.random.shuffle(self.train_inds)
        self.actual_train_inds = self.train_inds[:int(len(self.train_inds) / 4)]
        print('Train on {} examples'.format(len(self.actual_train_inds)))

    def train_generator(self, batch_size):
        return self._get_generator(self.game.all_objs[self.actual_train_inds],
                                   self.game.all_msgs[self.actual_train_inds],
                                   batch_size)

    def val_generator(self, batch_size):
        """ Used for evaluation. For each validation loop, randomly pick EVALUATION_RATIO * total_objects
        """
        split = int(EVALUATION_RATIO * len(self.all_indices))
        valid_ids = self.all_indices[self.valid_start: self.valid_start + split]
        self.valid_start += split

        # Reset valid_start if exceeding limit and reshuffle ids
        if self.valid_start >= len(self.all_indices):
            self.valid_start = 0
            np.random.shuffle(self.all_indices)

        # Take the validation data
        objs = self.game.all_objs[valid_ids]
        msgs = self.game.all_msgs[valid_ids]
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
