"""
This script takes a tf board folder structure like
exp
|__exp1
|    |__ run1
|    |__ run2
|    |__ run1
|    |__ run1
|    |__ run3
|    |__ ...
|    |__ runN
|
|__exp2
     |__ run1
     |__ run2
     |__ run3
     |__ ...
     |__ runN

And generate plots with shaded area for all
"""
import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from math import ceil

# The Tags that I care about
TAGS = ['ho_comm_acc', 'ho_speak_gr_acc', 'sp_comm_acc', 'sp_speak_gr_acc']

# Plot Config
NB_COL = 2
NB_ROW = ceil(len(TAGS) / NB_COL)


class Series:
    def __init__(self):
        self.values = []
        self.steps = []

    def add(self, step, val):
        """ Insert step and value. Maintain sorted w.r.t. steps """
        if len(self.steps) == 0:
            self.steps.append(step)
            self.values.append(val)
        else:
            for idx in reversed(range(len(self.steps))):
                if step > self.steps[idx]:
                    break
            else:
                idx = -1
            self.steps.insert(idx + 1, step)
            self.values.insert(idx + 1, val)

    def verify(self):
        for i in range(len(self.steps) - 1):
            assert self.steps[i] <= self.steps[i + 1]


def combine_series(series_list):
    """
    :param series_list: a list of `Series` assuming steps are aligned
    :return: steps, means, stds
    """
    step_sizes = [len(series.steps) for series in series_list]
    min_idx = np.argmin(step_sizes)
    steps = series_list[min_idx].steps

    # [nb_run, nb_steps]
    all_values = [series.values[:len(steps)] for series in series_list]
    means = np.mean(all_values, axis=0)
    stds = np.std(all_values, axis=0)
    return steps, means, stds


def parse_tb_event_file(event_file):
    data = {}
    for e in tf.compat.v1.train.summary_iterator(event_file):
        for v in e.summary.value:
            tag = v.tag.replace('/', '_')
            if tag in TAGS:
                if data.get(tag) is None:
                    data[tag] = Series()
                data[tag].add(step=e.step, val=v.simple_value)

    for tag in data:
        data[tag].verify()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    exp_names = os.listdir(args.exp_dir)
    print('We have {} experiments'.format(len(exp_names)))

    all_data = {}
    all_tags = []
    for exp_name in exp_names:
        all_data[exp_name] = []
        runs = os.listdir(os.path.join(args.exp_dir, exp_name))
        print('We have {} runs for {}'.format(len(runs), exp_name))
        for run in runs:
            event_file = os.listdir(os.path.join(args.exp_dir, exp_name, run))[0]
            run_data = parse_tb_event_file(os.path.join(args.exp_dir, exp_name, run, event_file))
            if len(run_data) == 0:
                continue
            all_tags = list(run_data.keys())
            all_data[exp_name].append(run_data)

    # Start plotting
    fig, axs = plt.subplots(NB_ROW, NB_COL, figsize=(4*NB_ROW, 5*NB_COL))
    for tag, ax in zip(all_tags, axs.reshape(-1)):
        for exp_name in exp_names:
            steps, means, stds = combine_series([run_data[tag] for run_data in all_data[exp_name]])
            line, = ax.plot(steps, means)
            line.set_label(exp_name)
            ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
        ax.legend()
        ax.set_xlabel('steps')
        ax.set_title(tag)
    fig.savefig(os.path.join(args.output_dir, 'output.png'))


if __name__ == '__main__':
    main()
