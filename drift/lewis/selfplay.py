from drift.lewis.core import LewisGame
from drift.lewis.pretrain import Dataset, eval_loop
from drift.lewis.linear import Speaker, Listener
import torch
from torch.distributions import Categorical

SPEAKER_CKPT = "./s_sl.pth"
LISTENER_CKPT = './l_sl.pth'
TRAIN_STEPS = 1000
BATCH_SIZE = 10
LOG_STEPS = 10
USE_GUMBEL = False  # If true use STE Gumbel


def main():
    speaker = Speaker.load(SPEAKER_CKPT)
    listener = Listener.load(LISTENER_CKPT)
    game = LewisGame(**speaker.env_config)
    dset = Dataset(game, 1)

    s_opt = torch.optim.Adam(lr=5e-4, params=speaker.parameters())
    l_opt = torch.optim.Adam(lr=5e-4, params=listener.parameters())

    for step in range(TRAIN_STEPS):
        if step % LOG_STEPS == 0:
            stats = eval_loop(dset.val_generator(1000), listener=listener,
                              speaker=speaker)
            logstr = ["epoch {}:".format(step)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
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
            rewards = (-l_logprobs).detach()
            s_logprobs = Categorical(s_logits).log_prob(msgs).sum(-1)
            reinforce = rewards * s_logprobs
            s_loss = (-reinforce).mean()

        s_opt.zero_grad()
        s_loss.backward()
        s_opt.step()


if __name__ == '__main__':
    main()
