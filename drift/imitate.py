""" One model imitate the other model """
import torch

from drift.pretrain import train_listener_batch, train_speaker_batch
from drift.core import eval_speaker_loop


def imitate_listener_batch(student, teacher, opt, msgs, temperature=0):
    """ Imitate teacher on this batch. If temperature > 0, it's imitate soft label.
        Else imitate argmax
    """
    # Generate target obj
    with torch.no_grad():
        oh_msgs = teacher.one_hot(msgs)
        teacher_logits = teacher.get_logits(oh_msgs)

    # Train with argmax
    if temperature == 0:
        objs = torch.argmax(teacher_logits, -1)
        train_listener_batch(student, opt, objs, msgs)

    else:
        soft_label = torch.nn.functional.softmax(teacher_logits / temperature, -1)
        student_logits = student.get_logits(oh_msgs)
        student_logprobs = torch.nn.functional.log_softmax(student_logits, -1)
        loss = -(soft_label * student_logprobs).sum(-1).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def imitate_speak_batch(student, teacher, opt, objs, temperature=0., use_sample=False, student_ctx=False):
    """ Imitate teacher on this batch. If temperature > 0, it's imitate soft label.
        Else imitate argmax
    """
    # Generate context msg
    with torch.no_grad():
        t_msgs = teacher.greedy(objs)
        context = student.greedy(objs) if student_ctx else t_msgs

    # Train with argmax
    if temperature == 0:
        train_speaker_batch(student, opt, objs, t_msgs)

    elif not use_sample:
        teacher_logits = teacher.get_logits(msgs=context, objs=objs)
        soft_label = torch.nn.functional.softmax(teacher_logits / temperature, -1)
        student_logits = student.get_logits(msgs=context, objs=objs)
        student_logprobs = torch.nn.functional.log_softmax(student_logits, -1)
        loss = -(soft_label * student_logprobs).sum(-1).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    else:
        teacher_logits = teacher.get_logits(msgs=context, objs=objs)
        msgs = torch.distributions.Categorical(logits=teacher_logits / temperature).sample()
        train_speaker_batch(student, opt, objs, msgs)


def listener_finetune(game, student_listener, max_steps, distilled_speaker, opt=None):
    l_opt = opt
    step = 0
    try:
        while True:
            if step >= max_steps:
                break

            # Generate the msg
            objs = game.random_sp_objs(50)
            with torch.no_grad():
                _, msgs = distilled_speaker.sample(objs)

            # Train for a batch
            train_listener_batch(student_listener, l_opt, objs, msgs)
            step += 1

    except KeyboardInterrupt:
        pass


def listener_imitate(game, student_listener, teacher_listener, max_steps, distilled_speaker, temperature=0.,
                     opt=None):
    l_opt = opt
    step = 0
    try:
        while True:
            if step >= max_steps:
                break

            # Generate the msg
            objs = game.random_sp_objs(50)
            with torch.no_grad():
                _, msgs = distilled_speaker.sample(objs)

            # Train for a batch
            imitate_listener_batch(student_listener, teacher_listener, l_opt, msgs, temperature)
            step += 1

    except KeyboardInterrupt:
        pass


def speaker_imitate(game, student_speaker, teacher_speaker, max_steps, temperature=0., use_sample=False,
                    student_ctx=True, with_eval_data=True, opt=None):
    s_opt = opt
    step = 0
    statss = []
    try:
        while True:
            if step >= max_steps:
                break

            # Generate batch with teacher listener
            objs = game.random_sp_objs(50)
            imitate_speak_batch(student_speaker, teacher_speaker, s_opt, objs, temperature, use_sample, student_ctx)
            if with_eval_data and step % 10 == 0:
                stats = eval_speaker_loop(game.get_generator(1000), student_speaker)[0]
                statss.append(stats)
            step += 1

    except KeyboardInterrupt:
        pass

    if with_eval_data:
        new_statss = {key: [stats[key] for stats in statss] for key in statss[0]}
        return new_statss
