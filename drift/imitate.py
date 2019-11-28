""" One model imitate the other model """
import torch

from drift.pretrain import train_listener_batch, train_speaker_batch


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


def listener_imitate(game, student_listener, teacher_listener, max_steps, temperature=0, distilled_speaker=None):
    l_opt = torch.optim.Adam(lr=1e-4, params=student_listener.parameters())
    step = 0
    try:
        while True:
            if step >= max_steps:
                break

            # Generate the msg
            objs = game.get_random_objs(50)
            if distilled_speaker is None:
                msgs = game.objs_to_msg(game.get_random_objs(50))
            else:
                with torch.no_grad():
                    _, msgs = distilled_speaker.sample(objs)

            # Train for a batch
            imitate_listener_batch(student_listener, teacher_listener, l_opt, msgs, temperature)
            step += 1

    except KeyboardInterrupt:
        pass


def speaker_imitate(game, student_speaker, teacher_speaker, max_steps, temperature=0., use_sample=False,
                    student_ctx=True):
    s_opt = torch.optim.Adam(lr=1e-4, params=student_speaker.parameters())
    step = 0
    try:
        while True:
            if step >= max_steps:
                break

            # Generate batch with teacher listener
            objs = game.get_random_objs(50)
            imitate_speak_batch(student_speaker, teacher_speaker, s_opt, objs, temperature, use_sample, student_ctx)
            step += 1

    except KeyboardInterrupt:
        pass
