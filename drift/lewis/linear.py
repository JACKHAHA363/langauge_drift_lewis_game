"""
2-layer linear model like in ICLR paper
"""
from drift.lewis.core import BaseSpeaker, BaseListener
import torch


class Speaker(BaseSpeaker):
    def __init__(self, env_config):
        super(Speaker, self).__init__(env_config)
        self.linear1 = torch.nn.Linear(self.env_config['p'] * self.env_config['t'], 200)
        self.linear2 = torch.nn.Linear(200, self.env_config['p'] * self.env_config['p'] * self.env_config['t'])

    def forward(self, objs):
        """ return [bsz, nb_prop, vocab_size] """
        oh_objs = self._one_hot(objs).float()
        logits = self.linear2(self.linear1(oh_objs))
        return logits.view(objs.shape[0], self.env_config['p'], -1)

    def _one_hot(self, objs):
        """ Make input a concatenation of one-hot
        :param objs [bsz, nb_props]
        :param oh_objs [bsz, nb_props * nb_types]
        """
        oh_objs = torch.LongTensor(size=[objs.shape[0], objs.shape[1], self.env_config['t']])
        oh_objs.zero_()
        oh_objs.scatter_(2, objs.unsqueeze(-1), 1)
        return oh_objs.view([objs.shape[0], -1])


class Listener(BaseListener):
    def __init__(self, env_config):
        super(Listener, self).__init__(env_config)
        self.emb = torch.nn.Embedding(self.env_config['p'] * self.env_config['t'], 200)
        self.linear = torch.nn.Linear(200, self.env_config['t'])

    def forward(self, msgs):
        """ return [bsz, nb_prop, type_size] """
        # [bsz, p, 200]
        embs = self.emb(msgs)
        logits = self.linear(embs)
        return logits



