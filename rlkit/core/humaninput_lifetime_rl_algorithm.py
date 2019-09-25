import abc
import csv
import os
import time

import gtimer as gt
from rlkit.core import logger
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import numpy as np

from os.path import join


class HumanInputLifetimeRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            # param not used but kept for consistency
            num_eval_steps_per_epoch,
            # this is really eval steps since eval = expl
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            human_input_interval=10,
            num_loops_before_human_input=10,
            rollout_env=None,
            **kwargs
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            **kwargs
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.evaluation_env = evaluation_env
        self.human_input_interval = human_input_interval
        self.rollout_env = rollout_env
        self.num_loops_before_human_input = num_loops_before_human_input

    def collect_rollout_gif(self, human_input_counter):

        agent = self.eval_data_collector._policy
        path_len = 100
        env = self.rollout_env
        o = env.reset()
        imgs = []
        for i in range(path_len):
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)

            img = env.render() #save=logger.get_snapshot_dir()+ '/gifs/%d.png' % i)
            imgs.append(img.getArray().transpose(1, 0, 2))
            time.sleep(0.1)
        time.sleep(0.3)

        # Add blacks frames to end
        for _ in range(5):
            imgs.append(imgs[-1])
        from array2gif import write_gif
        write_gif(imgs, logger.get_snapshot_dir() + '/train_%d.gif' % human_input_counter, fps=5)

    def make_plot(self, key):
        import matplotlib.pyplot as plt
        data = np.genfromtxt(join(logger.get_snapshot_dir(), 'validation_stats.csv'), delimiter=',', names=True)
        y = data[key]
        if y.shape:
            # only plot if we have meaningful (>1) validation stats
            plt.plot(np.arange(y.shape[0]), y)
            plt.savefig(join(logger.get_snapshot_dir(), 'plot_%s.png' % key))


    def set_radius(self, r):
        self.eval_data_collector._env.human_set_place_radius(r)
        self.rollout_env.human_set_place_radius(r)

    def get_human_radius_input(self, human_input_counter):
        correct_input = False
        while not correct_input:
            human_input = input("Input radius resource distance between 2 and %d (inclusive). Previously at %d: "
                                % ((self.rollout_env.grid_size - 2) * 2 - 1, self.rollout_env.place_radius()))
            try:
                value = int(human_input)
                if 2 <= value <= (self.rollout_env.grid_size - 2) * 2 - 1:
                    correct_input = True
            except:
                continue
        return value

    def get_human_epochs_input(self, num_epochs):
        correct_input = False
        while not correct_input:
            human_input = input("Input number of epochs to train for, or 'rest' to leave this radius permanently. Must be at least %d (%d out of %d epochs completed so far): "
                                % (self.validation_period, num_epochs, self.num_epochs))
            try:
                value = int(human_input)
                if value >= self.validation_period:
                    correct_input = True
            except:
                if human_input == 'rest':
                    value = self.num_epochs
                    correct_input = True
        return value

    def _train(self):

        cur_radius = 2
        self.set_radius(cur_radius)

        if self.min_num_steps_before_training > 0:
            init_eval_path = self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                continuing=False
            )
            self.replay_buffer.add_paths([init_eval_path])
            self.eval_data_collector.end_epoch(-1)

            if np.any(init_eval_path['terminals']):
                return

        done = False
        num_loops = 0
        # human-specified number of epochs to train for. initialize to number of initial loops before input
        human_num_epochs = self.num_loops_before_human_input
        # number of loops since last human input was collected
        num_loops_since_input = 0
        human_input_counter = 0
        human_inputs = []

        with open(join(logger.get_snapshot_dir(), 'human_inputs.csv'), 'a') as f:
            writer = csv.DictWriter(f, fieldnames=['radius', 'num_epochs'])
            writer.writeheader()
        while not done:
            print('num_since: %d, human_epoch: %d' % (num_loops_since_input, human_num_epochs))
            num_loops += 1
            num_loops_since_input += 1
            if hasattr(self.evaluation_env, 'health'):
                print(
                    'Steps: %d, health: %d' % (num_loops * self.num_expl_steps_per_train_loop, self.evaluation_env.health))
            new_eval_path = self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
                continuing=True
            )

            gt.stamp('exploration sampling', unique=False)

            self.replay_buffer.add_paths([new_eval_path])
            gt.stamp('data storing', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(
                    self.batch_size)
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)
            done = num_loops >= self.num_epochs

            print('Ending epoch')
            self._end_epoch(num_loops - 1, incl_expl=False)

            # Human input section
            if num_loops_since_input == human_num_epochs or done:
                num_loops_since_input = 0
                self.collect_rollout_gif(human_input_counter)
                for key in ['proportion_solved_total']:
                    self.make_plot(key)
                radius = self.get_human_radius_input(human_input_counter)
                self.set_radius(radius)
                human_num_epochs = self.get_human_epochs_input(num_loops)
                human_input_counter += 1
                human_inputs.append(radius)
                with open(join(logger.get_snapshot_dir(), 'human_inputs.csv'), 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=['radius', 'num_epochs'])
                    writer.writerow({'radius': radius, 'num_epochs': human_num_epochs})

                # TODO Plot training and validation curves here





