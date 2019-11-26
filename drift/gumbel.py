from drift import USE_GPU
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

BATCH_SIZE = 500


def selfplay_batch(game, gumbel_temperature, l_opt, listener, s_opt, speaker):
    """ Generate a batch and play """
    # Generate batch
    objs = game.get_random_objs(BATCH_SIZE)
    y, msgs = speaker.gumbel(objs, gumbel_temperature)

    # Get gradient to keep backprop to speaker
    oh_msgs = listener.one_hot(msgs)
    oh_msgs.requires_grad = True
    oh_msgs.grad = None
    l_logits = listener.get_logits(oh_msgs)

    # Train listener
    l_logprobs = Categorical(logits=l_logits).log_prob(objs)
    l_logprobs = l_logprobs.sum(-1)
    l_opt.zero_grad()
    (-l_logprobs.mean()).backward(retain_graph=True)
    l_opt.step()

    # Train Speaker
    s_opt.zero_grad()
    assert oh_msgs.grad is not None
    y.backward(oh_msgs.grad)
    s_opt.step()
