import torch

from drift.core import BaseSpeaker, BaseListener


class Speaker(BaseSpeaker):
    def __init__(self, env_config):
        super(Speaker, self).__init__(env_config)
        self.linear1 = torch.nn.Linear(self.env_config['p'] * self.env_config['t'], 200)
        self.linear2 = torch.nn.Linear(200, self.env_config['p'] * self.env_config['p'] * self.env_config['t'])

    def forward(self, objs):
        """ return [bsz, nb_prop, vocab_size] """
        oh_objs = self._one_hot(objs)
        logits = self.linear2(self.linear1(oh_objs))
        return logits.view(objs.shape[0], self.env_config['p'], -1)

    def _one_hot(self, objs):
        """ Make input a concatenation of one-hot
        :param objs [bsz, nb_props]
        :param oh_objs [bsz, nb_props * nb_types]
        """
        oh_objs = torch.Tensor(size=[objs.shape[0], objs.shape[1], self.env_config['t']])
        oh_objs = oh_objs.to(device=objs.device)
        oh_objs.zero_()
        oh_objs.scatter_(2, objs.unsqueeze(-1), 1)
        return oh_objs.view([objs.shape[0], -1])


class Listener(BaseListener):
    def __init__(self, env_config):
        super(Listener, self).__init__(env_config)
        self.linear1 = torch.nn.Linear(self.env_config['p'] * self.env_config['t'], 200)
        self.linear2 = torch.nn.Linear(200, self.env_config['t'])

    def forward(self, oh_msgs):
        """ return [bsz, nb_prop, type_size] """
        return self.linear2(self.linear1(oh_msgs))

    def one_hot(self, msgs):
        """
        :param msgs: [bsz, nb_props]
        :return: [bsz, nb_props, vocab_size]
        """
        oh_msgs = torch.Tensor(size=[msgs.shape[0], msgs.shape[1], self.env_config['p'] * self.env_config['t']])
        oh_msgs = oh_msgs.to(device=msgs.device)
        oh_msgs.zero_()
        oh_msgs.scatter_(2, msgs.unsqueeze(-1), 1)
        return oh_msgs