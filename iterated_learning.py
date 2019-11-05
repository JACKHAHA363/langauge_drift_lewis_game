"""
Try using reset https://arxiv.org/pdf/1906.02403.pdf
"""
import os
import argparse
from copy import deepcopy
import torch
import random
from shutil import rmtree
from tensorboardX import SummaryWriter
from drift.core import LewisGame, Dataset, eval_loop, get_comm_acc
from drift.gumbel import selfplay_batch
from drift.pretrain import listener_imitate, speaker_imitate
from drift import USE_GPU

STEPS = 400000
LOG_STEPS = 20


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt_dir', required=True, help='path to save/load ckpts')
    parser.add_argument('-generation_steps', type=int, default=2500, help='Reset one of the agent to the checkpoint '
                                                                          'of that steps before')
    parser.add_argument('-s_transmission_steps', type=int, default=1500, help='number of steps to transmit signal '
                                                                              'for speaker')
    parser.add_argument('-l_transmission_steps', type=int, default=1500, help='number of steps to transmit signal '
                                                                              'for listener')
    parser.add_argument('-logdir', required=True, help='path to tb log')
    parser.add_argument('-temperature', type=float, default=10, help='Initial temperature')
    parser.add_argument('-decay_rate', type=float, default=1., help='temperature decay rate. Default no decay')
    parser.add_argument('-min_temperature', type=float, default=1, help='Minimum temperature')
    parser.add_argument('-save_vocab_change', default=None, help='Path to save the vocab change results. '
                                                                 'If None not save')
    return parser.parse_args()


def _load_pretrained_agents(args):
    # Make sure it's valid ckpt dirs
    all_ckpts = os.listdir(args.ckpt_dir)
    assert 's0.pth' in all_ckpts
    assert 'l0.pth' in all_ckpts

    speaker = torch.load(os.path.join(args.ckpt_dir, 's0.pth'))
    if USE_GPU:
        speaker = speaker.cuda()
    s_opt = torch.optim.Adam(lr=5e-5, params=speaker.parameters())

    listener = torch.load(os.path.join(args.ckpt_dir, 'l0.pth'))
    if USE_GPU:
        listener = listener.cuda()
    l_opt = torch.optim.Adam(lr=5e-5, params=listener.parameters())
    return speaker, s_opt, listener, l_opt


def iteration_selfplay(args):
    """ Load checkpoints """
    # Load populations
    speaker, s_opt, listener, l_opt = _load_pretrained_agents(args)
    game = LewisGame(**speaker.env_config)
    dset = Dataset(game, 1)
    if os.path.exists(args.logdir):
        rmtree(args.logdir)
    writer = SummaryWriter(args.logdir)

    # Training
    temperature = args.temperature
    vocab_change_data = {'speak': [], 'listen': []}
    s_ckpt = deepcopy(speaker.state_dict())
    l_ckpt = deepcopy(listener.state_dict())
    try:
        for step in range(STEPS):
            # Train for a Batch
            speaker.train(True)
            listener.train(True)
            selfplay_batch(game, temperature, l_opt, listener, s_opt, speaker)
            temperature = max(args.min_temperature, temperature * args.decay_rate)

            # Check if randomly reset one of speaker or listener to previous ckpt
            if (step + 1) % args.generation_steps == 0:
                # Restore to old version and start transmission
                teacher_speaker = speaker.from_state_dict(speaker.env_config, speaker.state_dict())
                teacher_speaker.train(False)
                speaker.load_state_dict(s_ckpt)
                teacher_listener = listener.from_state_dict(listener.env_config, listener.state_dict())
                teacher_listener.train(False)
                listener.load_state_dict(l_ckpt)

                print('Start transmission')
                speaker_imitate(student_speaker=speaker,
                                teacher_speaker=teacher_speaker, max_steps=args.s_transmission_steps)
                listener_imitate(student_listener=listener,
                                 teacher_listener=teacher_listener, max_steps=args.l_transmission_steps)

                # Save for future student
                s_ckpt = deepcopy(speaker.state_dict())
                l_ckpt = deepcopy(listener.state_dict())

            # Eval and Logging
            if step % LOG_STEPS == 0:
                speaker.train(False)
                listener.train(False)
                stats, s_conf_mat, l_conf_mat = eval_loop(dset.val_generator(1000), listener=listener,
                                                          speaker=speaker, game=game)
                writer.add_image('s_conf_mat', s_conf_mat.unsqueeze(0), step)
                writer.add_image('l_conf_mat', l_conf_mat.unsqueeze(0), step)

                if args.save_vocab_change is not None:
                    vocab_change_data['speak'].append(s_conf_mat)
                    vocab_change_data['listen'].append(l_conf_mat)
                stats.update(get_comm_acc(dset.val_generator(1000), listener, speaker))
                stats['temp'] = temperature
                logstr = ["step {}:".format(step)]
                for name, val in stats.items():
                    logstr.append("{}: {:.4f}".format(name, val))
                    writer.add_scalar(name, val, step)
                writer.flush()
                print(' '.join(logstr))
                #if stats['comm_acc'] == 1.:
                #    stats['step'] = step
                #    break

    except KeyboardInterrupt:
        pass

    if args.save_vocab_change is not None:
        for key, val in vocab_change_data.items():
            vocab_change_data[key] = torch.stack(val)
        torch.save(vocab_change_data, 'zzz_vocab_data.pth')


if __name__ == '__main__':
    args = get_args()
    iteration_selfplay(args)
