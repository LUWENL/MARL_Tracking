import os.path
import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
from pprint import pprint
import matplotlib.pyplot as plt



def _t2n(x):
    return x.detach().cpu().numpy()


class TrackingRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the Tracking Scenario. See parent class for details."""

    def __init__(self, config):
        super(TrackingRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "MARL_TRACKING":
                    env_infos = {}
                    # for agent_id in range(self.num_agents):
                    #     idv_rews = []
                    #     for info in infos:
                    #         if 'individual_reward' in info[agent_id].keys():
                    #             idv_rews.append(info[agent_id]['individual_reward'])
                    #     agent_k = 'agent%i/individual_rewards' % agent_id
                    #     env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        print(self.buffer.share_obs[0].shape, share_obs.shape)

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError


        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps, for_paper_evaluation = False):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        sat_names  = self.eval_envs.envs[0].Sat_Name
        tar_names  = self.eval_envs.envs[0].Tar_Name
        sat_tar_angle = {sat_name: {tar_name: [] for tar_name in tar_names} for sat_name in sat_names}
        tar_track_status = {tar_name: np.zeros(self.episode_length, dtype=int) for tar_name in tar_names}
        task_id = self.eval_envs.envs[0].scenario_id
        if not os.path.exists(f"task{task_id}"):
            os.makedirs(f"task{task_id}")

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            if for_paper_evaluation:
                pprint(eval_infos[0])

                # angles
                for sat_name in sat_names:
                    for tar_name in tar_names:
                        angle = eval_infos[0][sat_name]["Angles"][tar_name]
                        sat_tar_angle[sat_name][tar_name].append(angle)

                # under tracking
                for tar_name in tar_names:
                    is_tracked = 0
                    if len(eval_infos[0][tar_name]['Being tracked']) > 0:
                        is_tracked = 1
                    tar_track_status[tar_name][eval_step] = is_tracked

            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        # angles
        for sat_name in sat_tar_angle.keys():
            fig, ax = plt.subplots(figsize=(12, 6))
            for tar_name, angle_list in sat_tar_angle[sat_name].items():
                ax.plot(range(self.episode_length), angle_list, label=tar_name, linewidth=1.5)

            ax.axhline(y=5, color='red', linestyle='--', linewidth=2)
            ax.text(self.episode_length - 1, 5, ' FOV half-angle',
                    verticalalignment='center', horizontalalignment='left',
                    color='red', fontsize=10)

            ax.set_xlabel('Timestep (s)', fontsize=13)
            ax.set_ylabel(f'Angle (deg) ({sat_name})', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', ncol=2, fontsize=15)
            ax.set_xlim(0, self.episode_length - 1)
            ax.set_ylim(bottom=0, top = 135)

            plt.tight_layout()
            plt.savefig(f'task{task_id}/{sat_name}_angle_timeline.pdf', dpi=500, bbox_inches='tight')
            plt.show()

        # is under tracking
        track_matrix = np.array([[tar_track_status[tar_name][step] for step in range(self.episode_length)] for tar_name in tar_names])
        cmap = plt.cm.colors.ListedColormap(['#DCDCDC', '#2E86AB'])  # Gray for untracked, blue for tracked
        bounds = [0, 0.5, 1]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(figsize=(30, 8))
        # im = ax.imshow(track_matrix, cmap=cmap, norm=norm, aspect='auto')
        im = ax.imshow(track_matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='none')

        # Axis labels and title
        ax.set_xlabel('Time Step', fontsize=13, fontweight='bold')
        ax.set_ylabel('Target Name', fontsize=13, fontweight='bold')

        # Y-axis (target names)
        ax.set_yticks(np.arange(len(tar_names)))
        ax.set_yticklabels(tar_names, fontsize=15)

        # X-axis (time steps) - Avoid overcrowding (show every 50 steps)
        tick_interval = 50  # Show tick every 50 steps (adjust based on needs)
        ax.set_xticks(np.arange(0,  self.episode_length, tick_interval))
        ax.set_xticklabels(np.arange(0,  self.episode_length, tick_interval), fontsize=13, rotation=45)

        # Add minor grid lines for better time step separation
        ax.set_xticks(np.arange(-0.5,  self.episode_length, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(tar_names), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.1)

        # Color bar legend
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Untracked (Gray)', 'Tracked (Blue)'], fontsize=12, fontweight='bold')

        # Add total tracked duration labels (right side of plot)
        total_tracked = track_matrix.sum(axis=1)

        # Add detailed stats for each target
        for i, tar in enumerate(tar_names):
            stats_text = f'Tracked: {int(total_tracked[i])}/{self.episode_length} ({total_tracked[i] / self.episode_length * 100:.1f}%)'
            ax.text( self.episode_length + 10, i, stats_text, va='center', ha='left',
                    fontweight='bold', fontsize=12, color='darkblue')

        # Adjust x-axis limit to fit stats labels
        ax.set_xlim(-0.5,  self.episode_length + 80)
        # plt.savefig(f'task{task_id}/tracking_state.pdf', dpi=500)
        plt.savefig(f'task{task_id}/tracking_state.png', dpi=500)

        # Adjust layout to prevent label cutoff
        # plt.tight_layout()
        plt.show()



        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        pass
        # """Visualize the env."""
        # envs = self.envs
        #
        # all_frames = []
        # for episode in range(self.all_args.render_episodes):
        #     obs = envs.reset()
        #     if self.all_args.save_gifs:
        #         image = envs.render('rgb_array')[0][0]
        #         all_frames.append(image)
        #     else:
        #         envs.render('human')
        #
        #     rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        #     masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        #
        #     episode_rewards = []
        #
        #     for step in range(self.episode_length):
        #         calc_start = time.time()
        #
        #         self.trainer.prep_rollout()
        #         action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
        #                                                      np.concatenate(rnn_states),
        #                                                      np.concatenate(masks),
        #                                                      deterministic=True)
        #         actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        #         rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        #
        #         if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
        #             for i in range(envs.action_space[0].shape):
        #                 uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
        #                 if i == 0:
        #                     actions_env = uc_actions_env
        #                 else:
        #                     actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        #         elif envs.action_space[0].__class__.__name__ == 'Discrete':
        #             actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
        #         else:
        #             raise NotImplementedError
        #
        #         # Obser reward and next obs
        #         obs, rewards, dones, infos = envs.step(actions_env)
        #         episode_rewards.append(rewards)
        #
        #         rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        #         masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        #         masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        #
        #         if self.all_args.save_gifs:
        #             image = envs.render('rgb_array')[0][0]
        #             all_frames.append(image)
        #             calc_end = time.time()
        #             elapsed = calc_end - calc_start
        #             if elapsed < self.all_args.ifi:
        #                 time.sleep(self.all_args.ifi - elapsed)
        #         else:
        #             envs.render('human')
        #
        #     print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
        #
        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
