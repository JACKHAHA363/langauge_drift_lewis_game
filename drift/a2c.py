import torch
import numpy as np
from torch.distributions import Categorical


# class ExponentialMovingAverager:
#     def __init__(self, init_mean, gamma=0.1):
#         self.mean = init_mean
#         self.num = 1
#         self.gamma = gamma

#     def update(self, value):
#         coef = self.gamma / (1 + self.num)
#         self.mean = self.mean * coef + (1 - coef) * value
#         self.num += 1

class ExponentialMovingAverager:
    def __init__(self, gamma=0.1):
        self.mean = np.zeros(1000000) - 100
        self.num = np.zeros(1000000)
        self.gamma = gamma

    def update(self, values, idxes):
        coef = self.gamma / (1 + self.num[idxes])
        self.mean[idxes] = self.mean[idxes] * coef + (1 - coef) * values
        self.num[idxes] += 1


def selfplay_batch(objs, l_opt, listener, s_opt, speaker, ema_reward=None):
    """ Use exponential reward (kinda depricated not working)
    :return updated average reward
    """
    # Generate batch
    idxes = objs[:,5]*100000+objs[:,4]*10000+objs[:,3]*1000+objs[:,2]*100+objs[:,1]*10+objs[:,0]
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
    values = rewards.numpy()
    # Compute reward average
    if ema_reward is not None:
        ema_reward.update(values,idxes)
    else:
        ema_reward = ExponentialMovingAverager()
        ema_reward.update(values,idxes)
    s_dist = Categorical(s_logits)
    s_logprobs = s_dist.log_prob(msgs).sum(-1)
    reinforce = (rewards - torch.tensor(ema_reward.mean[idxes])) * s_logprobs
    entropy = s_dist.entropy().sum(-1)
    s_opt.zero_grad()
    (-reinforce.mean() - 0.0001 * entropy.mean()).backward()
    s_opt.step()
    return ema_reward


def selfplay_batch_a2c(objs, game, l_opt, listener, s_opt, speaker, value_coef, ent_coef):
    """ Use a learnt value function """
    # Generate batch
    a2c_info = speaker.a2c(objs)
    oh_msgs = listener.one_hot(a2c_info['msgs'])
    l_logits = listener.get_logits(oh_msgs)

    # Train listener
    l_logprobs = Categorical(logits=l_logits).log_prob(objs)
    l_logprobs = l_logprobs.sum(-1)
    l_opt.zero_grad()
    (-l_logprobs.mean()).backward(retain_graph=True)
    l_opt.step()

    # Policy gradient
    rewards = l_logprobs.detach()
    v_loss = torch.mean((a2c_info['values'] - rewards[:, None]).pow(2))

    adv = (rewards[:, None] - a2c_info['values']).detach()
    reinforce = adv * a2c_info['logprobs']
    p_loss = -reinforce.mean()

    ent_loss = -a2c_info['ents'].mean()

    s_opt.zero_grad()
    (p_loss + value_coef * v_loss + ent_coef * ent_loss).backward()
    s_opt.step()
