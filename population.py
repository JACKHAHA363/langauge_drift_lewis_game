"""
(Close to getting depricated...)
Try using population
    1. Training against many listener and many speakers
"""
import os
import argparse
import random
import torch
import numpy as np
from shutil import rmtree
from tensorboardX import SummaryWriter
from drift.gumbel import selfplay_batch
from drift.evaluation import eval_loop
from drift.a2c import selfplay_batch_a2c
from drift import USE_GPU


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt_dir', required=True, help='path to save/load ckpts')
    parser.add_argument('-seed', default=None, type=int, help='The global seed')
    parser.add_argument('-logdir', required=True, help='path to tb log')
    parser.add_argument('-save_vocab_change', default=None, help='Path to save the vocab change results. '
                                                                 'If None not save')
    parser.add_argument('-steps', default=10000, type=int, help='Total training steps')
    parser.add_argument('-log_steps', default=100, type=int, help='Log frequency')
    parser.add_argument('-s_lr', default=1e-3, type=float, help='learning rate for speaker')
    parser.add_argument('-l_lr', default=1e-3, type=float, help='learning rate for listener')
    parser.add_argument('-batch_size', default=100, type=int, help='Batch size')
    parser.add_argument('-method', choices=['gumbel', 'a2c'], default='gumbel', help='Which way to train')

    # S2P
    parser.add_argument('-n', type=int, default=5, help='population size')

    # For gumbel
    parser.add_argument('-temperature', type=float, default=10, help='Initial temperature')
    parser.add_argument('-decay_rate', type=float, default=1., help='temperature decay rate. Default no decay')
    parser.add_argument('-min_temperature', type=float, default=1, help='Minimum temperature')

    # For a2c
    parser.add_argument('-v_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('-ent_coef', type=float, default=0.001, help='entropy reg coefficient')
    return parser.parse_args()


def _load_population_and_opts(args, s_ckpts, l_ckpts):
    s_and_opts = []
    l_and_opts = []
    for i in range(args.n):
        speaker = torch.load(os.path.join(args.ckpt_dir, s_ckpts[i]))
        if USE_GPU:
            speaker = speaker.cuda()
        s_opt = torch.optim.Adam(lr=args.s_lr, params=speaker.parameters())
        s_and_opts.append([speaker, s_opt])

        listener = torch.load(os.path.join(args.ckpt_dir, l_ckpts[i]))
        if USE_GPU:
            listener = listener.cuda()
        l_opt = torch.optim.Adam(lr=args.l_lr, params=listener.parameters())
        l_and_opts.append([listener, l_opt])
    return s_and_opts, l_and_opts


def population_selfplay(args):
    """ Load checkpoints """
    # Make sure it's valid ckpt dirs
    all_ckpts = os.listdir(args.ckpt_dir)
    s_ckpts = ['s{}.pth'.format(i) for i in range(args.n)]
    for s_ckpt in s_ckpts:
        assert s_ckpt in all_ckpts
    l_ckpts = ['l{}.pth'.format(i) for i in range(args.n)]
    for l_ckpt in l_ckpts:
        assert l_ckpt in all_ckpts

    # Load populations
    s_and_opts, l_and_opts = _load_population_and_opts(args, s_ckpts, l_ckpts)
    game = torch.load(os.path.join(args.ckpt_dir, 'game.pth'))
    if os.path.exists(args.logdir):
        rmtree(args.logdir)
    writer = SummaryWriter(args.logdir)

    # Training
    temperature = args.temperature
    vocab_change_data = {}
    try:
        for step in range(args.steps):
            # Randomly pick one pair
            speaker, s_opt = random.choice(s_and_opts)
            listener, l_opt = random.choice(l_and_opts)

            # Train for a Batch
            speaker.train(True)
            listener.train(True)
            objs = game.random_sp_objs(args.batch_size)
            if args.method == 'gumbel':
                selfplay_batch(objs, temperature, l_opt, listener, s_opt, speaker)
                temperature = max(args.min_temperature, temperature * args.decay_rate)
            elif args.method == 'a2c':
                selfplay_batch_a2c(objs, l_opt, listener, s_opt, speaker, args.v_coef, args.ent_coef)
            else:
                raise NotImplementedError

            # Eval and Logging
            if step % args.log_steps == 0:
                eval_loop(listener, speaker, game, writer, step, vocab_change_data)

    except KeyboardInterrupt:
        pass

    if args.save_vocab_change is not None:
        for key, val in vocab_change_data.items():
            vocab_change_data[key] = torch.stack(val)
        torch.save(vocab_change_data, args.save_vocab_change)

    if args.save_population is not None:
        for idx, ((speaker, _), (listener, _)) in enumerate(zip(s_and_opts, l_and_opts)):
            speaker.save(os.path.join(args.save_population, 's{}.pth'.format(idx)))
            listener.save(os.path.join(args.save_population, 'l{}.pth'.format(idx)))


if __name__ == '__main__':
    args = get_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    print('Train with:', args.method)
    population_selfplay(args)
