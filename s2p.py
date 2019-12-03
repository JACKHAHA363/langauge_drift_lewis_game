"""
Try using reset https://arxiv.org/pdf/1906.02403.pdf
"""
import os
import argparse
import torch
import numpy as np
from shutil import rmtree
from tensorboardX import SummaryWriter
from drift.evaluation import eval_loop
from drift.gumbel import selfplay_batch
from drift.a2c import selfplay_batch_a2c
from drift.pretrain import train_speaker_batch, train_listener_batch
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
    parser.add_argument('-supervise_freq', type=int, default=5, help='Perform one supervise step at this freq')

    # For gumbel
    parser.add_argument('-temperature', type=float, default=10, help='Initial temperature')
    parser.add_argument('-decay_rate', type=float, default=1., help='temperature decay rate. Default no decay')
    parser.add_argument('-min_temperature', type=float, default=1, help='Minimum temperature')

    # For a2c
    parser.add_argument('-v_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('-ent_coef', type=float, default=0.001, help='entropy reg coefficient')
    return parser.parse_args()


def _load_pretrained_agents(args):
    # Make sure it's valid ckpt dirs
    all_ckpts = os.listdir(args.ckpt_dir)
    assert 's0.pth' in all_ckpts
    assert 'l0.pth' in all_ckpts

    speaker = torch.load(os.path.join(args.ckpt_dir, 's0.pth'))
    if USE_GPU:
        speaker = speaker.cuda()
    s_opt = torch.optim.Adam(lr=args.s_lr, params=speaker.parameters())

    listener = torch.load(os.path.join(args.ckpt_dir, 'l0.pth'))
    if USE_GPU:
        listener = listener.cuda()
    l_opt = torch.optim.Adam(lr=args.l_lr, params=listener.parameters())
    return speaker, s_opt, listener, l_opt


def s2p(args):
    """ Load checkpoints """
    # Load populations
    speaker, s_opt, listener, l_opt = _load_pretrained_agents(args)
    teacher_speaker = speaker.from_state_dict(speaker.env_config, speaker.state_dict())
    teacher_listener = listener.from_state_dict(listener.env_config, listener.state_dict())
    if USE_GPU:
        teacher_speaker.cuda()
        teacher_listener.cuda()
    game = torch.load(os.path.join(args.ckpt_dir, 'game.pth'))
    game.info()
    if os.path.exists(args.logdir):
        rmtree(args.logdir)
    writer = SummaryWriter(args.logdir)

    # Training
    temperature = args.temperature
    vocab_change_data = {}
    try:
        for step in range(args.steps):
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

            # Perform one supervise steps
            if (step + 1) % args.supervise_freq == 0:
                batch_objs, batch_msgs = game.random_su_objs_msgs(args.batch_size)
                train_speaker_batch(speaker, s_opt, batch_objs, batch_msgs)
                train_listener_batch(listener, l_opt, batch_objs, batch_msgs)

            # Eval and Logging
            if step % args.log_steps == 0:
                eval_loop(listener, speaker, game, writer, step, vocab_change_data)

    except KeyboardInterrupt:
        pass

    if args.save_vocab_change is not None:
        for key, val in vocab_change_data.items():
            vocab_change_data[key] = torch.stack(val)
        torch.save(vocab_change_data, args.save_vocab_change)


if __name__ == '__main__':
    args = get_args()
    print('########## Config ###############')
    for key, val in args.__dict__.items():
        print('{}: {}'.format(key, val))
    print('#################################')
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    s2p(args)
