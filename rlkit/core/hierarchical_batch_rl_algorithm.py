import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import gtimer as gt
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector
from rlkit.torch.core import torch_ify


class HierarchicalBatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            low_replay_buffer: ReplayBuffer,
            high_replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_low_trains_per_train_loop,
            num_high_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            goal_std=1,

            viz_maps=False,
            **kwargs
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            # we don't use replay_buffer
            None,
            viz_maps,
            **kwargs
        )
        self.low_replay_buffer = low_replay_buffer
        self.high_replay_buffer = high_replay_buffer
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_low_trains_per_train_loop = num_low_trains_per_train_loop
        self.num_high_trains_per_train_loop = num_high_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.goal_std = goal_std
        self.viz_maps = viz_maps

    def _train(self):
        if self.min_num_steps_before_training > 0:
            low_init_expl_paths, high_init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.low_replay_buffer.add_paths(low_init_expl_paths)
            self.high_replay_buffer.add_paths(high_init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                low_new_expl_paths, high_new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                print('collected data')
                gt.stamp('exploration sampling', unique=False)

                self.low_replay_buffer.add_paths(low_new_expl_paths)
                self.high_replay_buffer.add_paths(high_new_expl_paths)
                print('added to replay buffer')
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_low_trains_per_train_loop):
                    low_train_data = self.low_replay_buffer.random_batch(
                        self.batch_size
                    )
                    self.trainer.low_train(low_train_data)
                print('low train')
                for _ in range(self.num_high_trains_per_train_loop):
                    high_train_data = self.high_replay_buffer.random_batch(
                        self.batch_size
                    )
                    high_train_data = self.relabel_data(high_train_data)
                    print('relabeled')
                    self.trainer.high_train(high_train_data)
                    print('high train')
                gt.stamp('training', unique=False)
                self.training_mode(False)
            print('epoch done')
            self._end_epoch(epoch)

    def relabel_data(self, data):
        """ HIRO off-policy correction """
        data = dict(data)
        for k, v in data.items():
            data[k] = v.copy()

        # generate 10 candidates, of which 8 are sampled from a gaussian around the empirical goal
        empirical_goal = data['next_observations'] - data['observations']
        gaussian_cands = np.random.randn(8, *data['observations'].shape) * self.goal_std + empirical_goal
        candidates = np.concatenate([np.expand_dims(empirical_goal, axis=0),
                                     np.expand_dims(data['actions'], axis=0),
                                     gaussian_cands],
                                    axis=0)
        candidates = np.transpose(candidates, (1, 0, 2))



        winning_cands = []
        for cands, obs, acs in zip(candidates, data['traj_obs'], data['traj_acs']):
            # looping over individual high level actions, corresponding to low level trajs
            dists = []
            for goal in cands:
                goal = goal.copy()
                dist = 0
                for ob, ac, next_ob in zip(obs, acs, obs[1:]):
                    ac_pol, _ = self.trainer.policy.get_action(np.hstack((ob, goal)))
                    diff = ac - ac_pol
                    dist += np.inner(diff, diff)
                    goal = self.trainer.setter.goal_transition(ob, goal, next_ob)
                dists.append(dist)
            winning_cands.append(np.array(dists).argmin())
        data['goals'] = candidates[np.arange(len(candidates)), winning_cands]
        return data

    def slow_relabel_data(self, data):
        """ HIRO off-policy correction """
        data = dict(data)
        for k, v in data.items():
            data[k] = v.copy()

        # generate 10 candidates, of which 8 are sampled from a gaussian around the empirical goal
        empirical_goal = data['next_observations'] - data['observations']
        gaussian_cands = np.random.randn(8, *data['observations'].shape) * self.goal_std + empirical_goal
        candidates = np.concatenate([np.expand_dims(empirical_goal, axis=0),
                                     np.expand_dims(data['actions'], axis=0),
                                     gaussian_cands],
                                    axis=0)
        candidates = np.transpose(candidates, (1, 0, 2))

        winning_cands = []
        for cands, obs, acs in zip(candidates, data['traj_obs'], data['traj_acs']):
            # looping over individual high level actions, corresponding to low level trajs
            dists = []
            for goal in cands:
                goal = goal.copy()
                dist = 0
                for ob, ac, next_ob in zip(obs, acs, obs[1:]):
                    ac_pol, _ = self.trainer.policy.get_action(np.hstack((ob, goal)))
                    diff = ac - ac_pol
                    dist += np.inner(diff, diff)
                    goal = self.trainer.setter.goal_transition(ob, goal, next_ob)
                dists.append(dist)
            winning_cands.append(np.array(dists).argmin())
        data['goals'] = candidates[np.arange(len(candidates)), winning_cands]
        return data

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _end_epoch(self, epoch, incl_eval=True):
        snapshot = self._get_snapshot()
        if self.viz_maps and epoch % self.viz_gap == 0:
            logger.save_viz(epoch, snapshot)
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving', unique=False)
        self._log_stats(epoch, incl_eval=incl_eval)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.low_replay_buffer.end_epoch(epoch)
        self.high_replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.low_replay_buffer.get_snapshot().items():
            snapshot['low_replay_buffer/' + k] = v
        for k, v in self.high_replay_buffer.get_snapshot().items():
            snapshot['high_replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch, incl_eval=True):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.low_replay_buffer.get_diagnostics(),
            prefix='low_replay_buffer/'
        )
        logger.record_dict(
            self.high_replay_buffer.get_diagnostics(),
            prefix='high_replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        low_expl_paths, high_expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(low_expl_paths),
                prefix='exploration/low/',
            )
            logger.record_dict(
                self.expl_env.get_diagnostics(high_expl_paths),
                prefix='exploration/high/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(low_expl_paths),
            prefix="exploration/low/",
        )
        logger.record_dict(
            eval_util.get_generic_path_information(high_expl_paths),
            prefix="exploration/high/",
        )

        if incl_eval:
            """
            Evaluation
            """
            logger.record_dict(
                self.eval_data_collector.get_diagnostics(),
                prefix='evaluation/',
            )
            low_eval_paths, high_eval_paths = self.eval_data_collector.get_epoch_paths()
            if hasattr(self.eval_env, 'get_diagnostics'):
                logger.record_dict(
                    self.eval_env.get_diagnostics(low_eval_paths),
                    prefix='evaluation/low/',
                )
                logger.record_dict(
                    self.eval_env.get_diagnostics(high_eval_paths),
                    prefix='evaluation/high/',
                )
            logger.record_dict(
                eval_util.get_generic_path_information(low_eval_paths),
                prefix="evaluation/low/",
            )
            logger.record_dict(
                eval_util.get_generic_path_information(high_eval_paths),
                prefix="evaluation/high/",
            )

        """
        Misc
        """
        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
