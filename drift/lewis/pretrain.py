import torch
from torch.distributions import Categorical
from drift.lewis.core import LewisGame, eval_loop, get_comm_acc
from drift.lewis.linear import Listener, Speaker

VAL_BATCH_SIZE = 1000
LOG_STEPS = 100


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


def train_batch(l_opt, listener, s_opt, speaker, objs, msgs):
    """ Perform a train step """
    s_logits = speaker(objs)
    s_logprobs = Categorical(logits=s_logits).log_prob(msgs)
    s_opt.zero_grad()
    (-s_logprobs.mean()).backward()
    s_opt.step()

    l_logits = listener(listener.one_hot(msgs))
    l_logprobs = Categorical(logits=l_logits).log_prob(objs)
    l_opt.zero_grad()
    (-l_logprobs.mean()).backward()
    l_opt.step()


def train(train_batch_size, train_steps, train_size):
    env_config = LewisGame.get_default_config()
    game = LewisGame(**env_config)
    dset = Dataset(game, train_size)

    speaker = Speaker(env_config)
    s_opt = torch.optim.Adam(lr=5e-4, params=speaker.parameters())
    listener = Listener(env_config)
    l_opt = torch.optim.Adam(lr=5e-4, params=listener.parameters())
    # writer = SummaryWriter('log_pretrain')
    step = 0
    while step < train_steps:
        for objs, msgs in dset.train_generator(train_batch_size):
            train_batch(l_opt, listener, s_opt, speaker, objs, msgs)
            if step % LOG_STEPS == 0:
                stats, _ = eval_loop(dset.val_generator(VAL_BATCH_SIZE), listener=listener,
                                     speaker=speaker)
                stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
                logstr = ["epoch {}:".format(step)]
                for name, val in stats.items():
                    logstr.append("{}: {:.4f}".format(name, val))
                    # writer.add_scalar(name, val, epoch)
                print(' '.join(logstr))
            step += 1

    stats, _ = eval_loop(dset.val_generator(VAL_BATCH_SIZE), listener=listener,
                         speaker=speaker)
    stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
    return speaker, listener, stats


if __name__ == '__main__':
    speaker, listener, stats = train(50, 100, 100)
    logstr = ["epoch {}:".format(100)]
    for name, val in stats.items():
        logstr.append("{}: {:.4f}".format(name, val))
        # writer.add_scalar(name, val, epoch)
    print(' '.join(logstr))
    speaker.save('s_sl.pth')
    listener.save('l_sl.pth')
