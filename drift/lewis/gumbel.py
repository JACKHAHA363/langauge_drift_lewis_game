from drift.lewis.core import LewisGame
from drift.lewis.pretrain import Dataset, eval_loop
from drift.lewis.linear import Speaker, Listener
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
GUMBEL_TEMP = 0.1  # Temperature for gumbel softmax
LOG_NAME = 'log_gumbel'


def get_comm_acc(val_generator, listener, speaker):
    corrects = 0
    total = 0
    for objs, _ in val_generator:
        with torch.no_grad():
            s_logits = speaker(objs)
            msgs = torch.argmax(s_logits, dim=-1)
            l_logits = listener(listener.one_hot(msgs))
            preds = torch.argmax(l_logits, dim=-1)
            corrects += (preds == objs).float().sum().item()
            total += objs.numel()
    return {'comm_acc': corrects / total}


def main():
    speaker = Speaker.load(SPEAKER_CKPT)
    listener = Listener.load(LISTENER_CKPT)
    game = LewisGame(**speaker.env_config)
    dset = Dataset(game, 1)

    s_opt = torch.optim.Adam(lr=5e-5, params=speaker.parameters())
    l_opt = torch.optim.Adam(lr=5e-5, params=listener.parameters())
    if os.path.exists(LOG_NAME):
        rmtree(LOG_NAME)
    writer = SummaryWriter(LOG_NAME)

    for step in range(TRAIN_STEPS):
        if step % LOG_STEPS == 0:
            stats, s_conf_mat = eval_loop(dset.val_generator(1000), listener=listener,
                                          speaker=speaker)
            writer.add_image('s_conf_mat', s_conf_mat.unsqueeze(0), step)
            stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
            logstr = ["epoch {}:".format(step)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
                writer.add_scalar(name, val, step)
            print(' '.join(logstr))

        # Generate batch
        objs = game.get_random_objs(BATCH_SIZE)
        s_logits = speaker(objs)

        y = torch.nn.functional.softmax(s_logits / GUMBEL_TEMP, dim=-1)
        g = torch.distributions.Gumbel(loc=0, scale=1).sample(y.shape)
        msgs = torch.argmax(torch.log(y) + g, dim=-1)

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
        y.backward(oh_msgs.grad)
        s_opt.step()


if __name__ == '__main__':
    main()
