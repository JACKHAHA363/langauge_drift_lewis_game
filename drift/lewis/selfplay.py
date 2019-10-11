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
BATCH_SIZE = 1000
LOG_STEPS = 10
USE_GUMBEL = False  # If true use STE Gumbel
LOG_NAME = 'log_selfplay'


def get_comm_acc(val_generator, listener, speaker):
    corrects = 0
    total = 0
    for objs, _ in val_generator:
        with torch.no_grad():
            s_logits = speaker(objs)
            msgs = torch.argmax(s_logits, dim=-1)
            l_logits = listener(msgs)
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
            stats = eval_loop(dset.val_generator(1000), listener=listener,
                              speaker=speaker)
            stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
            logstr = ["epoch {}:".format(step)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
                writer.add_scalar(name, val, step)
            print(' '.join(logstr))

        # Generate batch
        objs = game.get_random_objs(BATCH_SIZE)
        s_logits = speaker(objs)

        if USE_GUMBEL:
            raise NotImplementedError
        else:
            msgs = Categorical(logits=s_logits).sample()
        l_logits = listener(msgs)

        # Train listener
        l_logprobs = Categorical(logits=l_logits).log_prob(objs)
        l_logprobs = l_logprobs.sum(-1)
        l_opt.zero_grad()
        (-l_logprobs.mean()).backward()
        l_opt.step()

        # Train Speaker
        if USE_GUMBEL:
            raise NotImplementedError

        # Policy gradient
        else:
            rewards = l_logprobs.detach()
            s_logprobs = Categorical(s_logits).log_prob(msgs).sum(-1)
            reinforce = rewards * s_logprobs
            s_loss = (-reinforce).mean()

        s_opt.zero_grad()
        s_loss.backward()
        s_opt.step()


if __name__ == '__main__':
    main()
