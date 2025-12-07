# MARL_Tracking

### :bulb: Active Continuous Tracking of Moving Targets by Multiple Satellites Based on Multi-Agent Reinforcement Learning


<p align="center">
<img src="problem_statement.png" width="1500px" height="550px" />
</p>

### keyword
Multi-agent Reinforcement Learning, Moving Target Tracking, Satellite Attitude control, Intelligent systems.

> Abstract: Active continuous tracking of moving targets by multiple agile satellites is crucial for time-sensitive missions such as situational awareness and dynamic surveillance, yet it confronts challenges including real-time decision-making, decentralized coordination, and the maintenance of tracking continuity.
Traditional "scheduling first, tracking second" frameworks are plagued by tracking interruptions caused by target-switching delays and exhibit heavy reliance on high-performance inter-satellite communication, which limits their adaptability to complex space environments.
To address these issues, this paper proposes a multi-agent reinforcement learning (MARL)-based framework integrated with a trajectory prediction module for active continuous tracking of moving targets.
Specifically, the tracking problem is formulated as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP), enabling each satellite agent to make real-time decisions based solely on local observations without relying on the transmission of global state information.
An efficient target trajectory prediction module—equipped with an improved neural network architecture that combines a Gated Embedding Layer and Multi-Head Wavelet Attention—is designed to forecast target motion dynamics, providing state information to support precise tracking adjustments.
The framework adopts an end-to-end direct torque control strategy, where a pre-trained Multi-Agent Proximal Policy Optimization (MAPPO)-based policy network maps local observations directly to actuator commands with millisecond-level inference speed.
Comparative experiments with representative tracking frameworks across three scenarios of varying complexity demonstrate that the proposed framework leverages the trajectory prediction module and learning-based controller to significantly improve continuous tracking performance—outperforming other frameworks in tracking continuity—while eliminating the high inter-satellite communication requirements and scheduling-induced time overhead, thus fully meeting the strict real-time demands of time-sensitive target tracking missions.


## 🛰️ Our Implementations
### MARL_Tracking Env for MARL training/testing
[Environment Implementation Code](onpolicy/envs/marltracking).
### Target Trajectory Prediction: Perceptual Wavelet Transform based Flight Trajectory Prediction (PWTFTP)
[PWTFTP Implementation Code](onpolicy/envs/models/TTP_module).
### Three Tracking Scenarios
| Scenario   | Num of Satellites | Num of Targets | Duration |
|------------|------------|---------|----------|
| Scenario 1 | 2          | 2       | 10 mins  | 
| Scenario 2 | 4          | 3       | 12 mins  |
| Scenario 3 | 5          | 4       | 15 mins  |


## 🤖 Getting started
### 🏃 For Training
#### <a id="Step1">Step 1</a>: Install the on-policy package according to [Here](https://github.com/marlbenchmark/on-policy/tree/main?tab=readme-ov-file#2-installation).
#### <a id="Step2">Step 2</a>: Modify algorithm parameters in [METADATA](onpolicy/envs/marltracking/metadata.py).
```.bash
"mode": 'train'
"scenario_id": 1/2/3,
```
#### <a id="Step3">Step 3</a>: Run the script in [task1_configuration](onpolicy/task1_configuration) / [task2_configuration](onpolicy/task2_configuration) / [task3_configuration](onpolicy/task3_configuration)  based on your own mission requirements.
```.bash
# Take task1 as an example
--env_name MARL_TRACKING --algorithm_name mappo --experiment_name TAES_31actions --scenario_name task1 --num_agents 2 --seed 1 --n_training_threads 1 --n_rollout_threads 2
--num_mini_batch 1 --episode_length 600 --num_env_steps 5000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "MARL_TRACKING" --user_name "xxxxx"
```

### 🏃 For Testing
#### <a id="Step1">Step 1</a>: Install the on-policy package according to [Here](https://github.com/marlbenchmark/on-policy/tree/main?tab=readme-ov-file#2-installation).
#### <a id="Step2">Step 2</a>: Modify algorithm parameters in [METADATA](onpolicy/envs/marltracking/metadata.py).
```.bash
"mode": 'test'
"scenario_id": 1/2/3,
```
#### <a id="Step3">Step 3</a>: Run the script in [task1_configuration](onpolicy/task1_configuration) / [task2_configuration](onpolicy/task2_configuration) / [task3_configuration](onpolicy/task3_configuration)  based on your own mission requirements.
```.bash
# Take task1 as an example
--env_name MARL_TRACKING --algorithm_name mappo --experiment_name TAES_test --scenario_name task1 --num_agents 2 --seed 1 --n_training_threads 1 --n_rollout_threads 1
--num_mini_batch 1 --episode_length 600 --num_env_steps 5000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "MARL_TRACKING" --user_name "xxxxx"
--use_eval True --model_dir ../../scripts/results/MARL_TRACKING/task1/mappo/TAES_31actions/xxxxx
```


## 👍 Acknowledgements
This project is built on the codebases of [MAPPO](https://github.com/marlbenchmark/on-policy) and [WTFTP](https://github.com/MusDev7/wtftp-model).

