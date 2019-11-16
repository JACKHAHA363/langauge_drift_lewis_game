import torch
import torch.nn.functional as F

from drift.core import BaseSpeaker, BaseListener
from drift import GUMBEL_DIST


class Speaker(BaseSpeaker):
    def __init__(self, env_config):
        super(Speaker, self).__init__(env_config)
        self.linear1 = torch.nn.Linear(self.env_config['p'] * self.env_config['t'], 200,
                                       bias=False)
        self.linear2 = torch.nn.Linear(200, self.env_config['p'] * self.env_config['p'] * self.env_config['t'],
                                       bias=False)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.linear1.weight, std=0.1)
        torch.nn.init.normal_(self.linear2.weight, std=0.1)

    def greedy(self, objs):
        logits = self.get_logits(objs)
        return torch.argmax(logits, -1)

    def gumbel(self, objs, temperature=1):
        logits = self.get_logits(objs)
        logprobs = F.log_softmax(logits, dim=-1)
        g = GUMBEL_DIST.sample(logits.shape)
        y = F.softmax((g + logprobs) / temperature, dim=-1)
        msgs = torch.argmax(y, dim=-1)
        return y, msgs

    def sample(self, objs):
        logits = self.get_logits(objs)
        dist = torch.distributions.Categorical(logits=logits)
        msgs = dist.sample()
        logprobs = dist.log_prob(msgs)
        return logprobs, msgs

    def get_logits(self, objs, msgs=None):
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
        self.linear1 = torch.nn.Linear(self.env_config['p'] * self.env_config['t'], 200,
                                       bias=False)
        self.linear2 = torch.nn.Linear(200, self.env_config['t'], bias=False)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.linear1.weight, std=0.1)
        torch.nn.init.normal_(self.linear2.weight, std=0.1)

    def get_logits(self, oh_msgs):
        return self.linear2(self.linear1(oh_msgs))
