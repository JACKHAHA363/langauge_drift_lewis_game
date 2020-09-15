"""
Lewis Signal Game
"""
import torch
from torch.nn.functional import softmax
from drift import USE_GPU
from drift.utils import _obj_prob_to_msg_prob, increment_2d_matrix


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
    def greedy(self, objs):
        """ Just for evaluate
        :return msg [bsz, len]
        """
        raise NotImplementedError

    def sample(self, objs):
        """ Return: logprobs, sample_msg
        :return logprobs [bsz, len]
                sample_msg [bsz, len]
        """
        raise NotImplementedError

    def gumbel(self, objs, temperature=1):
        """ Return y, gumbel_msg
        :return y: [bsz, len, vocab_size]
                    F.softmax(gumbel_noise + logprobs) / temperature)
                gumbel_msg: [bsz, len]
        """
        raise NotImplementedError

    def get_logits(self, objs, msgs):
        """ Compute log P(msgs | objs)
        :param objs: [bsz, nb_props]
        :param msgs: [bsz, nb_props]
        :return: logits. [bsz, nb_props, vocab_size]
        """
        raise NotImplementedError

    def a2c(self, objs):
        """ The method for decoding during policy gradient
        :return a2c_info. A dict contains:
                msgs: [bsz, len]
                logprobs: [bsz, len]
                ents: [bsz, len]
                values [bsz, len]
        """
        raise NotImplementedError


class BaseListener(Agent):
    """ Listener """

    def get_logits(self, oh_msgs):
        """ Method for testing
        :param oh_msgs: [bsz, nb_props, vocab_size]
        :param objs: [bsz, nb_props]
        :return: logits: [bsz, nb_props, nb_types]
        """
        raise NotImplementedError

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


"""
Evaluation Utils
"""
def eval_comm_loop(generator, listener, speaker):
    """ Return the communication accuracy on give generator """
    corrects = 0
    total = 0
    vocab_size = listener.env_config['p'] * listener.env_config['t']
    l_conf_mat_gr_msg = torch.zeros([vocab_size, vocab_size])
    if USE_GPU:
        l_conf_mat_gr_msg = l_conf_mat_gr_msg.cuda()
    for objs, _ in generator:
        with torch.no_grad():
            # Communicate
            gr_msgs = speaker.greedy(objs)
            l_logits = listener.get_logits(listener.one_hot(gr_msgs))
            preds = torch.argmax(l_logits, dim=-1)
            corrects += (preds == objs).float().sum().item()
            total += objs.numel()

            # Listener acc on speaker msg
            l_probs = softmax(l_logits, dim=-1)
            increment_2d_matrix(l_conf_mat_gr_msg, gr_msgs.view(-1),
                                _obj_prob_to_msg_prob(l_probs).view([-1, vocab_size]))
    acc_gr_msg = (l_conf_mat_gr_msg.diag().sum() / l_conf_mat_gr_msg.sum()).item()
    l_conf_mat_gr_msg /= (1e-32 + torch.sum(l_conf_mat_gr_msg, -1, keepdim=True))
    return {'comm_acc': corrects / total, 'listen/acc_gr_msg': acc_gr_msg}, l_conf_mat_gr_msg


def eval_speaker_loop(generator, speaker):
    """ Return stats and conf matrix """
    tf_corrects = 0
    gr_corrects = 0
    total = 0
    ent = 0
    nb_batch = 0
    vocab_size = speaker.env_config['p'] * speaker.env_config['t']
    conf_mat = torch.zeros([vocab_size, vocab_size])
    if USE_GPU:
        conf_mat = conf_mat.cuda()

    for objs, msgs in generator:
        nb_batch += 1
        total += msgs.numel()

        with torch.no_grad():
            # Teacher Forcing
            logits = speaker.get_logits(objs=objs, msgs=msgs)
            pred = torch.argmax(logits, dim=-1)
            tf_corrects += (pred == msgs).float().sum().item()

            # Greedy
            gr_msgs = speaker.greedy(objs=objs)
            gr_corrects += (gr_msgs == msgs).float().sum().item()
            gr_logits = speaker.get_logits(objs=objs, msgs=gr_msgs)
            gr_probs = softmax(gr_logits, dim=-1)
            increment_2d_matrix(conf_mat, msgs.view(-1), gr_probs.view(-1, vocab_size))
            ent += -(gr_probs * torch.log(gr_probs + 1e-32)).mean().item()

    conf_mat /= (1e-32 + torch.sum(conf_mat, -1, keepdim=True))
    return {'speak/tf_acc': tf_corrects / total,
            'speak/gr_acc': gr_corrects / total,
            'speak/ent': ent / nb_batch}, conf_mat


def eval_listener_loop(generator, listener):
    """ Return stats and conf mat """
    corrects = 0
    total = 0
    ent = 0
    nb_batch = 0
    vocab_size = listener.env_config['p'] * listener.env_config['t']
    conf_mat = torch.zeros([vocab_size, vocab_size])
    if USE_GPU:
        conf_mat = conf_mat.cuda()
    for objs, msgs in generator:
        nb_batch += 1
        total += objs.numel()
        with torch.no_grad():
            logits = listener.get_logits(listener.one_hot(msgs))
            pred = torch.argmax(logits, dim=-1)
            corrects += (pred == objs).float().sum().item()

            probs = softmax(logits, dim=-1)
            increment_2d_matrix(conf_mat, msgs.view(-1), _obj_prob_to_msg_prob(probs).view([-1, vocab_size]))
            ent += -(probs * torch.log(probs + 1e-32)).mean().item()
    return {'listen/acc': corrects / total, 'listen/ent': ent / nb_batch}, conf_mat
