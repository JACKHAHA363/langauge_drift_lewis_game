"""
2-layer linear model like in ICLR paper
"""
from drift.lewis.core import BaseSpeaker, BaseListener
import torch


class Speaker(BaseSpeaker):
    def __init__(self, env_config):
        super(Speaker, self).__init__()
        self.env_config = env_config
        self.linear1 = torch.nn.Linear(self.env_config['p'] * self.env_config['t'], 200)
        self.linear2 = torch.nn.Linear(200, self.env_config['p'] * self.env_config['p'] * self.env_config['t'])

    def output_msg(self, objs):
        """ Predict by taking argmax """
        logits = self.forward(objs)
        return torch.argmax(logits, -1)

    def get_msg_logprobs(self, objs, msgs):
        logits = self.forward(objs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(msgs)

    def forward(self, objs):
        """ return [bsz, nb_prop, vocab_size] """
        oh_objs = self._one_hot(objs).float()
        logits = self.linear2(self.linear1(oh_objs))
        return logits.view(objs.shape[0], self.env_config['p'])

    def _one_hot(self, objs):
        """ Make input a concatenation of one-hot
        :param objs [bsz, nb_props]
        :param oh_objs [bsz, nb_props * nb_types]
        """
        oh_objs = torch.LongTensor(size=[objs.shape[0], objs.shape[1], self.env_config['t']])
        oh_objs.zero_()
        oh_objs.scatter_(2, objs.unsqueeze(-1), 1)
        return oh_objs.view([objs.shape[0], -1])

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


if __name__ == '__main__':
    from drift.lewis.core import LewisGame
    env_config = LewisGame.get_default_config()
    game = LewisGame(**env_config)
    s = Speaker(env_config)
    objs = game.get_random_objs(20)
    with torch.no_grad():
        msgs = s.output_msg(objs)
        print(msgs)

    # train
    msgs_logprobs = s.get_msg_logprobs(objs, msgs)
    print(msgs_logprobs.shape)



