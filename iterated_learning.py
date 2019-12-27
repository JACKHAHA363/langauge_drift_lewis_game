"""
Try using reset https://arxiv.org/pdf/1906.02403.pdf
"""
import os
import argparse
from copy import deepcopy
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from shutil import rmtree
from tensorboardX import SummaryWriter
from drift.evaluation import eval_loop
from drift.core import eval_speaker_loop, eval_comm_loop, eval_listener_loop
from drift.gumbel import selfplay_batch
from drift.a2c import selfplay_batch_a2c
from drift.imitate import listener_imitate, speaker_imitate, listener_finetune
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

    # For transmission
    parser.add_argument('-init_weight', action='store_true', help='Use the ckpt weights')
    parser.add_argument('-distill_temperature', type=float, default=0, help='If 0 fit with argmax, else use '
                                                                            'soft label with that temperature')
    parser.add_argument('-student_ctx', action='store', help='Whether or not to use student as context '
                                                             'during speaker recurrent distillation')
    parser.add_argument('-generation_steps', type=int, default=2500, help='Reset one of the agent to the checkpoint '
                                                                          'of that steps before')
    parser.add_argument('-s_transmission_steps', type=int, default=1500, help='number of steps to transmit signal '
                                                                              'for speaker. '
                                                                              'If negative -1, no transmission')
    parser.add_argument('-l_transmission_steps', type=int, default=1500, help='number of steps to transmit signal '
                                                                              'for listener.'
                                                                              'If negative -1, no transmission')
    parser.add_argument('-s_use_sample', action='store_true')
    parser.add_argument('-l_finetune', action='store_true')

    # For gumbel
    parser.add_argument('-temperature', type=float, default=10, help='Initial temperature')
    parser.add_argument('-decay_rate', type=float, default=1., help='temperature decay rate. Default no decay')
    parser.add_argument('-min_temperature', type=float, default=1, help='Minimum temperature')

    # For a2c
    parser.add_argument('-v_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('-ent_coef', type=float, default=0.001, help='entropy reg coefficient')
    return parser.parse_args()


def plot_distill_change(vocab_size, final_s_conf_mat, student_s_conf_mat,
                        teacher_s_conf_mat, distill_temperature):
    """ Plot each teacher, student, and final """
    final_s_conf_mat = final_s_conf_mat.cpu()
    student_s_conf_mat = student_s_conf_mat.cpu()
    teacher_s_conf_mat = teacher_s_conf_mat.cpu()

    # Distort teacher with temperature
    teacher_s_conf_mat = torch.softmax(torch.log(teacher_s_conf_mat) / distill_temperature, dim=-1)
    NB_PLOT_PER_ROW = 5
    WORDS_TO_PLOT = [i for i in range(0, vocab_size, 1)]
    NB_ROW = math.ceil(len(WORDS_TO_PLOT) / NB_PLOT_PER_ROW)
    fig, axs = plt.subplots(NB_ROW, NB_PLOT_PER_ROW, figsize=(int(100 / NB_ROW),
                                                              int(100 / NB_PLOT_PER_ROW)))
    for word_id, ax in zip(range(vocab_size), axs.reshape(-1)):
        ax.plot(student_s_conf_mat[word_id].numpy(), '--', label='student',)
        ax.plot(teacher_s_conf_mat[word_id].numpy(), '--', label='teacher')
        ax.plot(final_s_conf_mat[word_id].numpy(), '--', label='final')
        ax.plot([word_id, word_id], [-0.1, 1.1], 'r-', label='true word')
        ax.legend()
        ax.set_title('word {}'.format(word_id))
        ax.set_ylim([-0.1, 1.1])

    return fig


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


def iteration_selfplay(args):
    """ Load checkpoints """
    # Load populations
    speaker, s_opt, listener, l_opt = _load_pretrained_agents(args)
    teacher_speaker = speaker.from_state_dict(speaker.env_config, speaker.state_dict())
    teacher_listener = listener.from_state_dict(listener.env_config, listener.state_dict())
    if USE_GPU:
        teacher_speaker.cuda()
        teacher_listener.cuda()
    s_ckpt = deepcopy(speaker.state_dict())
    l_ckpt = deepcopy(listener.state_dict())
    game = torch.load(os.path.join(args.ckpt_dir, 'game.pth'))
    if USE_GPU:
        game.cuda()
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

            # Check if randomly reset one of speaker or listener to previous ckpt
            if (step + 1) % args.generation_steps == 0:
                # Restore to old version and start transmission
                teacher_speaker.load_state_dict(speaker.state_dict())
                teacher_speaker.train(False)
                if args.s_transmission_steps >= 0:
                    speaker.load_state_dict(s_ckpt)
                teacher_listener.load_state_dict(listener.state_dict())
                teacher_listener.train(False)
                if args.l_transmission_steps >= 0:
                    listener.load_state_dict(l_ckpt)

                print('Start transmission')
                # Randomly pick some word for initialization
                _, student_s_conf_mat = eval_speaker_loop(game.get_generator(1000), speaker)
                _, teacher_s_conf_mat = eval_speaker_loop(game.get_generator(1000), teacher_speaker)
                if args.s_transmission_steps >= 0:
                    speaker_imitate(game=game, student_speaker=speaker, teacher_speaker=teacher_speaker,
                                    max_steps=args.s_transmission_steps, temperature=args.distill_temperature,
                                    use_sample=args.s_use_sample, student_ctx=args.student_ctx)

                # Distill using distilled speaker msg
                if args.l_transmission_steps >= 0:
                    if args.l_finetune:
                        print('Finetune listener!')
                        listener_finetune(game=game, student_listener=listener, max_steps=args.l_transmission_steps,
                                          distilled_speaker=speaker)
                    else:
                        print('Distill listener')
                        listener_imitate(game=game, student_listener=listener, teacher_listener=teacher_listener,
                                         max_steps=args.l_transmission_steps, temperature=args.distill_temperature,
                                         distilled_speaker=speaker)

                _, final_s_conf_mat = eval_speaker_loop(game.get_generator(1000), speaker)

                img = plot_distill_change(game.vocab_size, final_s_conf_mat=final_s_conf_mat,
                                          student_s_conf_mat=student_s_conf_mat,
                                          teacher_s_conf_mat=teacher_s_conf_mat,
                                          distill_temperature=args.distill_temperature)
                img.savefig(os.path.join(args.logdir, 'distill_{}.png'.format(step)))

                # Save for future student if do not use initial weight
                if not args.init_weight:
                    s_ckpt = deepcopy(speaker.state_dict())
                    l_ckpt = deepcopy(listener.state_dict())

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
    iteration_selfplay(args)
