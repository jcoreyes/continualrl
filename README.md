# Ecological RL
Under review as a conference paper at ICLR 2020

## Terminology
The following is a mapping from terminology used in the codebase to that used in the paper, where it differs:
 - `axe` in the code corresponds to the "tool-making task" in the paper
 - `deer` in the code corresponds to the "hunting task" in the paper
 - `waypoint` in the code refers to the "subgoal reward" in the paper

## Experiments by Figure
 - Figures 3 (Effect of Dynamic Environment on Episodic and Non-Episodic Learning) and 4 (Dynamic Ablations)
   - [experiment script directory](experiments/continual/dynamic_static)
 - Figure 5 (Shaping Methods for Episodic and Non-Episodic Learning)
   - [experiment script directory](experiments/continual/env_shaping/distance_increasing)
 - Figure 6 (Learned Behavior on Tool-Making Task)
   - [script used to generate figure](data/scripts/gen_validation_rollout_gifs_heatmaps.py)
 - Figure 7 (Human-Guided Environment Shaping)
   - [experiment script](experiments/continual/env_shaping/distance_increasing/axe/tool_dqn_human.py)
 - Figure 8 (Shaping Methods for Walled Tool-Making Task)
   - [experiment script](experiments/continual/env_shaping/env_vs_reward/wall/tool_dqn_wall_train.py)
 - Figures 9 and 10 (State Visitation Counts for WalledTool-Making Task)
   - [Jupyter notebook used to generate figure](data/scripts/gen_heatmaps.ipynb)
 - Figure 11 (Learned Behavior on Hunting Task)
   - [script used to generate figure](data/scripts/gen_validation_rollout_gifs_heatmaps.py)
   
## Experiment Workflow
The experiments require a set of validation environments for performance evaluation. Paired with each experiment script
is a script called `gen_validation_envs.py`, which outputs a Python pickle file containing these validation
environments, the path to which can be fed in directly to the experiment script as its `validation_envs_pkl` argument
in the `algorithm_kwargs` dictionary found in each experiment script.