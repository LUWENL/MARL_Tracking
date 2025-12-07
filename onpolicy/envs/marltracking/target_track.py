import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .models.TTP_module.MusDev7_wtftp_model.TTP_net import TTP_net
from .metadata import METADATA
from .models.satellite_model import Satellite
from .models.target_model import Target
from .models.model_utils import str2datetime, next_dt


class MARL_TrackEnv(gym.Env):
    metadata = {}

    def __init__(self, args):

        scenario_name = args.scenario_name
        num_agents = args.num_agents

        # get the id of scenarios
        self.scenario_id = str(METADATA['scenario_id'])
        assert self.scenario_id == scenario_name[-1], "Scenario ID {}!={} in METADATA.py should be identical with it in Script Config".format(self.scenario_id, scenario_name[-1])

        self.is_discrete = METADATA['is_discrete']

        if self.scenario_id == '1':
            from .task1 import Config1 as Config
        elif self.scenario_id == '2':
            from .task2 import Config2 as Config
        elif self.scenario_id == '3':
            from .task3 import Config3 as Config
        else:
            assert False, "Check the scenario id"

        assert Config['N_satellite'] == num_agents, "N_sat {}!={} in Task Config should be identical with it in Script Config".format(Config['N_satellite'], num_agents)

        self.N_satellite = Config['N_satellite']
        self.N_target = Config['N_target']
        self.Sat_Name = Config['Sat_Name']
        self.Tar_Name = Config['Tar_Name']
        self.satellite_fault = Config['satellite_fault']
        self.half_angle = Config['payload_half_angle']

        self.regions = []
        self.satellites = []
        self.targets = []

        # for evaluation
        self.fitness_list = []
        self.consumed_time_list = []

        # time
        self.time = 0
        self.start_datetime = str2datetime(METADATA['start_datetime'])
        self.end_datetime = str2datetime(Config['end_datetime'])
        self.time_length = int((self.end_datetime - self.start_datetime).total_seconds())

        # agent and target
        # id starts from 0
        for sat_id in range(self.N_satellite):
            self.satellites.append(
                Satellite(id=sat_id, name=self.Sat_Name[sat_id], start_datetime=self.start_datetime, scenario_id=self.scenario_id,
                          effectiveness_loss=self.satellite_fault[sat_id][:3], additive_bias=self.satellite_fault[sat_id][-3:], half_angle=self.half_angle[sat_id])
            )

        # id starts from 0
        # Flight Trajectory Prediction Module
        self.ttp_net = TTP_net(
            model_path="D:\My Program\Pycharm Projects\on-policy-main\onpolicy\envs\marltracking\models\TTP_module\MusDev7_wtftp_model\pretrained_models\PWTFTP.pt")
        self.ttp_net.load_model()

        for tar_id in range(self.N_target):
            self.targets.append(
                Target(id=tar_id, name=self.Tar_Name[tar_id], start_datetime=self.start_datetime, time_length=self.time_length, scenario_id=self.scenario_id, TTP_net=self.ttp_net))

        # action and observation
        self.discrete_action_space = True
        self.max_torque = 0.5
        self.torque1 = 0.035
        self.torque2 = 0.015
        self.torque3 = 0.0075
        self.torque4 = 0.0035


        self._action_to_torques = {
            0: np.array([0, 0, 0]),

            1: np.array([self.max_torque, 0, 0]),
            2: np.array([-self.max_torque, 0, 0]),
            3: np.array([0, self.max_torque, 0]),
            4: np.array([0, -self.max_torque, 0]),
            5: np.array([0, 0, self.max_torque]),
            6: np.array([0, 0, -self.max_torque]),

            7: np.array([self.torque1, 0, 0]),
            8: np.array([-self.torque1, 0, 0]),
            9: np.array([0, self.torque1, 0]),
            10: np.array([0, -self.torque1, 0]),
            11: np.array([0, 0, self.torque1]),
            12: np.array([0, 0, -self.torque1]),

            13: np.array([self.torque2, 0, 0]),
            14: np.array([-self.torque2, 0, 0]),
            15: np.array([0, self.torque2, 0]),
            16: np.array([0, -self.torque2, 0]),
            17: np.array([0, 0, self.torque2]),
            18: np.array([0, 0, -self.torque2]),

            19: np.array([self.torque3, 0, 0]),
            20: np.array([-self.torque3, 0, 0]),
            21: np.array([0, self.torque3, 0]),
            22: np.array([0, -self.torque3, 0]),
            23: np.array([0, 0, self.torque3]),
            24: np.array([0, 0, -self.torque3]),

            25: np.array([self.torque4, 0, 0]),
            26: np.array([-self.torque4, 0, 0]),
            27: np.array([0, self.torque4, 0]),
            28: np.array([0, -self.torque4, 0]),
            29: np.array([0, 0, self.torque4]),
            30: np.array([0, 0, -self.torque4]),


        }

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0

        # detailed action and obs config
        self.max_torque = 0.5
        quaternion_limit = np.array([1.0, 1.0, 1.0, 1.0])
        self.max_omega = self.satellites[0].max_omega
        # max_omega_for_clip = np.array([METADATA['max_omega_for_clip'] for i in range(3)])
        max_omega = np.array([METADATA['max_omega'] for i in range(3)])

        # el_high_limit = np.array([0.5 for i in range(3)])
        # el_low_limit = np.array([0 for i in range(3)])
        # ua_high_limit = np.array([0.05 for i in range(3)])
        # ua_low_limit = np.array([-0.05 for i in range(3)])

        # self.obs_high = np.array(np.concatenate([np.tile(quaternion_limit, self.N_target), max_omega, el_high_limit, ua_high_limit]), dtype=np.float32)
        # self.obs_low = np.array(np.concatenate([np.tile(-quaternion_limit, self.N_target), -max_omega, el_low_limit, ua_low_limit]), dtype=np.float32)
        self.obs_high = np.array(np.concatenate([np.tile(quaternion_limit, self.N_target), max_omega]), dtype=np.float32)
        self.obs_low = np.array(np.concatenate([np.tile(-quaternion_limit, self.N_target), -max_omega]), dtype=np.float32)

        for i in range(self.N_satellite):
            total_action_space = []

            if self.discrete_action_space:
                u_action_space = spaces.Discrete(31)
            else:
                u_action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(4,), dtype=np.float32)
            total_action_space.append(u_action_space)

            # # total action space
            # if len(total_action_space) > 1:
            #     # all action spaces are discrete, so simplify to MultiDiscrete action space
            #     if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
            #         act_space = MultiDiscrete(
            #             [[0, act_space.n - 1] for act_space in total_action_space])
            #     else:
            #         act_space = spaces.Tuple(total_action_space)
            #     self.action_space.append(act_space)
            # else:
            #     self.action_space.append(total_action_space[0])
            self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = 4 * self.N_target + 3
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=self.obs_low, high=self.obs_high, shape=(obs_dim,), dtype=np.float32))

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.N_satellite)]

    def step(self, actions):

        self.time += 1

        # update targets
        for i in range(self.N_target):
            self.targets[i].update()

        observation_list = []
        reward_list = []
        done_list = []
        info_dict = {}

        # update satellites
        for i in range(self.N_satellite):
            torques = self._action_to_torques[np.where(actions[i] == 1)[0][0]]
            # if i == 0:
            # print(torques)

            # update the satellite
            self.satellites[i].update(torques)

            s_reward = self.satellites[i].satellite_reward(targets=self.targets)
            reward_list.append(s_reward)

            done = (self.satellites[i].datetime == self.end_datetime)
            observation = self._get_obs(self.satellites[i])
            s_info = self._get_info(self.satellites[i], entity_type='satellite')

            observation_list.append(observation)
            done_list.append(done)

            info_dict[self.satellites[i].name] = s_info
            info_dict[self.satellites[i].name]['s_reward'] = s_reward

        for j in range(self.N_target):
            if len(self.targets[j].under_tracking) > 0:
                t_reward = 15 + 5 * (len(self.targets[j].under_tracking) - 1)
            else:
                t_reward = 0
            reward_list.append(t_reward)

            t_info = self._get_info(self.targets[j], entity_type='target')
            info_dict[self.targets[j].name] = t_info
            info_dict[self.targets[j].name]['t_reward'] = t_reward

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_list)

        # if shared_reward:
        if True:
            reward_list = [[reward]] * self.N_satellite

        # print(reward_list)

        return np.array(observation_list).reshape([self.N_satellite, 4 * self.N_target + 3]), reward_list, done_list, info_dict

    def reset(self, seed=None, options=None):

        # We need the following line to seed self.np_random
        super().reset(seed=METADATA['seed'])

        self.time = 0
        # reset target
        for i in range(self.N_target):
            # self.targets[i].reset(self.start_datetime, scenario_id=scenario_id, id=target_ids[i])
            self.targets[i].reset(self.start_datetime)

        # reset the satellite
        for i in range(self.N_satellite):
            self.satellites[i].reset(self.start_datetime)

        observation_list = []
        info_dict = {}
        for i in range(self.N_satellite):
            observation = self._get_obs(self.satellites[i])
            s_info = self._get_info(self.satellites[i], entity_type='satellite')

            observation_list.append(observation)
            info_dict[self.satellites[i].name] = s_info

        for j in range(self.N_target):
            t_info = self._get_info(self.targets[j], entity_type='target')
            info_dict[self.targets[j].name] = t_info

        # print(np.array(observation_list).reshape([self.N_satellite, 15]))

        # return np.array(observation_list).reshape([self.N_satellite,  5 * self.N_target + 3 + 2 * 3]), info_dict
        return np.array(observation_list).reshape([self.N_satellite, 4 * self.N_target + 3])

    def _get_obs(self, satellite):

        return satellite.observation(targets=self.targets)

    def _get_info(self, entity, entity_type='satellite'):

        # [0] is used for choose the 1-th target

        if entity_type == 'satellite':
            info = {
                "Current quaternion": entity.quaternion0,
                "Is tracking": entity.is_tracking,
                "Angles": entity.angles,
            }
        elif entity_type == 'target':
            info = {
                "Being tracked": entity.under_tracking
            }
        else:
            info = []

        return info

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass
