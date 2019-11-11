"""
Copy this and modify it to run
"""
import argparse
import os
from os.path import dirname, join
import random
import subprocess
import sys
import string
import logging
import drift
from itertools import product

LOGGER = logging.getLogger('random_hparam')
LOGGER.setLevel(logging.INFO)
LOGGER.handlers = [logging.StreamHandler(stream=sys.stdout)]

TRAIN_FILE = join(dirname(dirname(drift.__file__)), 'iterated_learning.py')

HP_RANGES = {
    'ckpt_dir': "",
    'distill_temperature': [0, 0.5, 1],
    'generation_steps': [1000, 2000, 3000],
    's_transmission_steps': [400, 600, 800],
    'l_transmission_steps': [400, 600, 800],
}


def get_hparam_generator(hparam_ranges):
    """ From the dictionary generate next hyparameter and save_dir
    :return hparam, exp_name
    """
    def generator():
        iteratable_keys = [key for key, r in hparam_ranges.items() if isinstance(r, list)]
        non_iteratable_keys = [key for key, r in hparam_ranges.items() if not isinstance(r, list)]
        for iteratable_values in product(*[hparam_ranges[k] for k in iteratable_keys]):
            hparams = dict()
            for idx, key in enumerate(iteratable_keys):
                hparams[key] = iteratable_values[idx]
            exp_name = '_'.join(['{}{}'.format(k, v) for k, v in hparams.items()])
            for k in non_iteratable_keys:
                hparams[k] = hparam_ranges[k]
            yield hparams, exp_name
    return generator()


def random_id():
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(8))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_dir', required=True, default=None,
                        help='Path to exp')
    parser.add_argument('-nb_workers', default=1, type=int,
                        help="nb of task run cocurrently")
    return parser.parse_args()


def main(args):
    exp_dir = args.exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    hparam_generator = get_hparam_generator(HP_RANGES)

    try:
        while True:
            ps = []
            base_cmd = ['python', TRAIN_FILE]
            for i in range(args.nb_workers):
                hparams, exp_name = hparam_generator.__next__()
                save_dir = os.path.join(exp_dir, exp_name)
                cmd = base_cmd + ['-logdir', save_dir]
                for key, val in hparams.items():
                    cmd += ['-{}'.format(key), str(val)]
                print(' '.join(cmd))

                LOGGER.info('#############')
                LOGGER.info(exp_name)
                LOGGER.info(cmd)
                LOGGER.info('#############')
                p = subprocess.Popen(cmd)
                ps.append(p)

            # Wait for it
            while True:
                ps_status = [p.poll() for p in ps]
                if all([x is not None for x in ps_status]):
                    break
            print('Finish {} jobs'.format(args.nb_workers))

    except StopIteration:
        pass


if __name__ == '__main__':
    main(get_args())
