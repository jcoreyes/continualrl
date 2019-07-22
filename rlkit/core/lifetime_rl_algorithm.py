import abc

import gtimer as gt
from rlkit.core import logger
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import numpy as np


class LifetimeRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
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
            # TODO below two params unnec?
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.exploration_env = exploration_env

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_path = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                continuing=False
            )
            self.replay_buffer.add_paths([init_expl_path])
            self.expl_data_collector.end_epoch(-1)

            if np.any(init_expl_path['terminals']):
                return

        done = False
        num_loops = 0
        while not done:
            num_loops += 1
            print('Steps: %d, health: %d' % (
            num_loops * self.num_expl_steps_per_train_loop, self.exploration_env.health))
            new_expl_path = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
                continuing=True
            )

            gt.stamp('exploration sampling', unique=False)

            self.replay_buffer.add_paths([new_expl_path])
            gt.stamp('data storing', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(
                    self.batch_size)
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)
            done = np.any(new_expl_path['terminals'])

            total_infos = {}
            for info in new_expl_path['env_infos']:
                for k, v in info:
                    total_infos[k] = total_infos.get(k, 0) + v
            logger.record_dict(
                total_infos,
                prefix='mined/'
            )

            print('Ending epoch')
            self._end_epoch(num_loops - 1, incl_eval=False)
