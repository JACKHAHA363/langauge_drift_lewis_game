import torch
from torch.distributions import Categorical
from drift.lewis.core import LewisGame
from drift.lewis.linear import Listener, Speaker

TRAIN_SIZE = 100
TRAIN_BATCH_SIZE = 5
VAL_BATCH_SIZE = 1000
NB_EPOCHS = 100
LOG_STEPS = 1


class Dataset:
    def __init__(self, game, train_size):
        assert isinstance(game, LewisGame)
        self.train_objs = game.get_random_objs(train_size)
        self.train_msgs = game.objs_to_msg(self.train_objs)
        self.train_size = train_size
        self.game = game

    def train_generator(self, batch_size):
        return self._get_generator(self.train_objs, self.train_msgs, batch_size)

    def val_generator(self, batch_size):
        # Randomly sample from all objects
        indices = torch.randint(len(self.game.all_objs), [5000]).long()
        objs = self.game.all_objs[indices]
        msgs = self.game.all_msgs[indices]
        return self._get_generator(objs, msgs, batch_size)

    @staticmethod
    def _get_generator(objs, msgs, batch_size):
        start = 0
        while start < len(objs):
            yield objs[start: start + batch_size], msgs[start: start + batch_size]
            start += batch_size


def train_loop(l_opt, listener, s_opt, speaker, train_generator):
    """ Perform a train step """
    for train_objs, train_msgs in train_generator:
        s_logits = speaker(train_objs)
        s_logprobs = Categorical(logits=s_logits).log_prob(train_msgs)
        s_opt.zero_grad()
        (-s_logprobs.mean()).backward()
        s_opt.step()

        l_logits = listener(train_msgs)
        l_logprobs = Categorical(logits=l_logits).log_prob(train_objs)
        l_opt.zero_grad()
        (-l_logprobs.mean()).backward()
        l_opt.step()


def eval_loop(val_generator, listener, speaker):
    l_corrects = 0
    l_total = 0
    s_corrects = 0
    s_total = 0
    for objs, msgs in val_generator:
        with torch.no_grad():
            l_logits = listener(msgs)
            l_pred = torch.argmax(l_logits, dim=-1)
            l_corrects += (l_pred == objs).float().sum().item()
            l_total += objs.numel()

            s_logits = speaker(objs)
            s_pred = torch.argmax(s_logits, dim=-1)
            s_corrects += (s_pred == msgs).float().sum().item()
            s_total += msgs.numel()

    return {'l_acc': l_corrects / l_total, 's_acc': s_corrects / s_total}


def main():
    env_config = LewisGame.get_default_config()
    game = LewisGame(**env_config)
    dset = Dataset(game, TRAIN_SIZE)

    speaker = Speaker(env_config)
    s_opt = torch.optim.Adam(lr=5e-4, params=speaker.parameters())
    listener = Listener(env_config)
    l_opt = torch.optim.Adam(lr=5e-4, params=listener.parameters())
    # writer = SummaryWriter('log_pretrain')
    for epoch in range(NB_EPOCHS):
        if epoch % LOG_STEPS == 0:
            stats = eval_loop(dset.val_generator(VAL_BATCH_SIZE), listener=listener,
                              speaker=speaker)
            logstr = ["epoch {}:".format(epoch)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
                # writer.add_scalar(name, val, epoch)
            print(' '.join(logstr))

        train_loop(l_opt, listener, s_opt, speaker, dset.train_generator(TRAIN_BATCH_SIZE))

    speaker.save('s_sl.pth')
    listener.save('l_sl.pth')


if __name__ == '__main__':
    main()
