from torch.distributions import Categorical
from drift.core import LewisGame
from drift.linear import Speaker, Listener


def test_speaker():
    env_config = LewisGame.get_default_config()
    game = LewisGame(**env_config)
    s = Speaker(env_config)
    l = Listener(env_config)
    objs = game.get_random_objs(20)
    msgs_logits = s.get_logits(objs)
    print(msgs_logits.shape)

    msgs = Categorical(logits=msgs_logits).sample()
    obj_logits = l.get_logits(l.one_hot(msgs))
    reconst = Categorical(logits=obj_logits).sample()
    print(obj_logits.shape)


if __name__ == '__main__':
    test_speaker()
