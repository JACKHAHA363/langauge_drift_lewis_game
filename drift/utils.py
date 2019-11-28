"""
Some utils
"""
import time

import torch


def timeit(info):
    def decorator(func):
        def wraped_func(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print("{}: {:.4f} sec".format(info, time.time() - start))
            return res
        return wraped_func
    return decorator


def _obj_prob_to_msg_prob(obj_probs):
    """
    :param obj_probs: [nb_obj, nb_type, nb_value]
    :return: [nb_obj, nb_type, nb_type * nb_value]
    """
    nb_obj, nb_type, nb_value = obj_probs.shape[0], obj_probs.shape[1], obj_probs.shape[2]
    result = torch.zeros([nb_obj, nb_type, nb_type * nb_value])
    result = result.to(device=obj_probs.device)
    for i in range(nb_type):
        start = i * nb_value
        end = (i+1) * nb_value
        result[:, i, start:end] = obj_probs[:, i]
    return result


def increment_2d_matrix(mat, row_id, row_updates):
    """
    :param row_id: [nb_indices]
    :param row_updates: [nb_indices, nb_col]
    """
    nb_row, nb_col = mat.shape

    # Build indices [nb_indices * nb_col]
    col = torch.arange(0, nb_col)
    indices = row_id[:, None] * nb_col + col[None, :]
    indices = indices.view(-1)

    mat.put_(indices, row_updates.view(-1), accumulate=True)


def combine_generator(generator_lists):
    """ Return a generator that will go through the list of generators """
    for gen in generator_lists:
        for stuff in gen:
            yield stuff
