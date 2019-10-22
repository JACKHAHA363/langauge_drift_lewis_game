from drift.core import LewisGame, get_comm_acc, eval_loop, Dataset
from drift.linear import Speaker, Listener
import torch
import os
from shutil import rmtree
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

SPEAKER_CKPT = "./s_sl.pth"
LISTENER_CKPT = './l_sl.pth'
TRAIN_STEPS = 10000
BATCH_SIZE = 500
LOG_STEPS = 50
LOG_NAME = 'log_pg'


class ExponentialMovingAverager:
    def __init__(self, init_mean, gamma=0.1):
        self.mean = init_mean
        self.num = 1
        self.gamma = gamma

    def update(self, value):
        coef = self.gamma / (1 + self.num)
        self.mean = self.mean * coef + (1 - coef) * value
        self.num += 1


def selfplay_batch(game, l_opt, listener, s_opt, speaker, ema_reward=None):
    """ Use exponential reward
    :return updated average reward
    """
    # Generate batch
    objs = game.get_random_objs(BATCH_SIZE)
    s_logits = speaker(objs)
    msgs = Categorical(logits=s_logits).sample()
    oh_msgs = listener.one_hot(msgs)
    l_logits = listener(oh_msgs)

    # Train listener
    l_logprobs = Categorical(logits=l_logits).log_prob(objs)
    l_logprobs = l_logprobs.sum(-1)
    l_opt.zero_grad()
    (-l_logprobs.mean()).backward(retain_graph=True)
    l_opt.step()
    # Policy gradient
    rewards = l_logprobs.detach()
    rewards_mean = rewards.mean().item()
    # Compute reward average
    if ema_reward is not None:
        ema_reward.update(rewards_mean)
    else:
        ema_reward = ExponentialMovingAverager(rewards_mean)
    s_dist = Categorical(s_logits)
    s_logprobs = s_dist.log_prob(msgs).sum(-1)
    reinforce = (rewards - ema_reward.mean) * s_logprobs
    entropy = s_dist.entropy().sum(-1)
    s_opt.zero_grad()
    (-reinforce.mean() - 0.0001 * entropy.mean()).backward()
    s_opt.step()
    return ema_reward

