from drift import USE_GPU
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

BATCH_SIZE = 500

# Generate the distribution for sampling
loc = torch.tensor(0.)
scale = torch.tensor(1.)
if USE_GPU:
    GUMBEL_DIST = torch.distributions.Gumbel(loc=loc.cuda(), scale=scale.cuda())
else:
    GUMBEL_DIST = torch.distributions.Gumbel(loc=loc, scale=scale)


def selfplay_batch(game, gumbel_temperature, l_opt, listener, s_opt, speaker):
    """ Generate a batch and play """
    # Generate batch
    objs = game.get_random_objs(BATCH_SIZE)
    s_logits = speaker(objs)
    s_logprob = F.log_softmax(s_logits, dim=-1)
    g = GUMBEL_DIST.sample(s_logits.shape)
    y = F.softmax((g + s_logprob) / gumbel_temperature, dim=-1)
    msgs = torch.argmax(y, dim=-1)
    # Get gradient to keep backprop to speaker
    oh_msgs = listener.one_hot(msgs)
    oh_msgs.requires_grad = True
    oh_msgs.grad = None
    l_logits = listener(oh_msgs)

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
