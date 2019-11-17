import torch

from drift.arch.linear import Listener as LinearListener, Speaker as LinearSpeaker


class Listener(LinearListener):
    def __init__(self, env_config):
        super(Listener, self).__init__(env_config)
        self.dropout = torch.nn.Dropout()

    def get_logits(self, oh_msgs):
        return self.linear2(self.dropout(self.linear1(oh_msgs)))


class Speaker(LinearSpeaker):
    def __init__(self, env_config):
        super(Speaker, self).__init__(env_config)
        self.dropout = torch.nn.Dropout()

    def get_logits(self, objs, msgs=None):
        oh_objs = self._one_hot(objs)
        logits = self.linear2(self.dropout(self.linear1(oh_objs)))
        return logits.view(objs.shape[0], self.env_config['p'], -1)
