"""
Track the change of symbol accuracy in gumbel selfplay.
python track_vocab_change.py -speaker [s_path] -listener [l_path] -out vocab_change.pth
vocab_data is a dictionary.
vocab_data['sdata' | 'ldata'] is the accuracy matrix of shape [NB_STEPS, VOCAB_SIZE]
"""
import torch
from drift.core import eval_loop, Dataset, LewisGame
from drift.gumbel import selfplay_batch
from tqdm import tqdm
import argparse


def main(args):
    speaker = torch.load(args.speaker)
    listener =torch.load(args.listener)

    game = LewisGame(**speaker.env_config)
    dset = Dataset(game, 1)

    s_opt = torch.optim.Adam(lr=5e-5, params=speaker.parameters())
    l_opt = torch.optim.Adam(lr=5e-5, params=listener.parameters())

    sdata = []
    ldata = []
    for step in tqdm(range(1000)):
        if step % args.log_step == 0:
            _, s_conf_mat, l_conf_mat = eval_loop(dset.val_generator(1000), listener=listener,
                                                  speaker=speaker, game=game)
            sdata.append(torch.diag(s_conf_mat))
            ldata.append(torch.diag(l_conf_mat))
        selfplay_batch(game, 1, l_opt, listener, s_opt, speaker)

    # [NB_data, vocab_size]
    sdata = torch.stack(sdata)
    ldata = torch.stack(ldata)
    torch.save({'sdata': sdata, 'ldata': ldata}, args.out)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-speaker', required=True, help='path to speaker')
    parser.add_argument('-listener', required=True, help='path to listener')
    parser.add_argument('-log_step', default=5, type=int, help='log frequency')
    parser.add_argument('-out', default='vocab_change.pth')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
