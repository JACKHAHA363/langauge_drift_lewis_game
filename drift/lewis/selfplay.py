from drift.lewis.core import LewisGame
from drift.lewis.pretrain import Dataset
from drift.lewis.linear import Speaker, Listener
from
import torch

SPEAKER_CKPT = "./s_sl.pth"
LISTENER_CKPT = './l_sl.pth'

def main():
    speaker = Speaker.load(SPEAKER_CKPT)
    listener = Listener.load(SPEAKER_CKPT)
    game = LewisGame(**speaker.env_config)
    dset = Dataset(game, 1)

    s_opt = torch.optim.Adam(lr=5e-4, params=speaker.parameters())
    l_opt = torch.optim.Adam(lr=5e-4, params=listener.parameters())
    writer = SummaryWriter('log_pretrain')
    for epoch in range(NB_EPOCHS):
        if epoch % LOG_STEPS == 0:
            stats = eval_loop(dset.val_generator(VAL_BATCH_SIZE), listener=listener,
                              speaker=speaker)
            logstr = ["epoch {}:".format(epoch)]
            for name, val in stats.items():
                logstr.append("{}: {:.4f}".format(name, val))
                writer.add_scalar(name, val, epoch)
            print(' '.join(logstr))

        train_loop(l_opt, listener, s_opt, speaker, dset.train_generator(TRAIN_BATCH_SIZE))

    speaker.save('s_sl.pth')
    listener.save('l_sl.pth')


if __name__ == '__main__':
    main()
