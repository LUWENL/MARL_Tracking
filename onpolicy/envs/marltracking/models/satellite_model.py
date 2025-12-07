import os
from audioop import error

import numpy as np
from .model_utils import next_dt, angle_between_vectors
from ..metadata import METADATA
from enum import Enum
from numpy import (cross, degrees, dot)
from .attitude_maneuver import *
from .dynamics_and_kinematics import *


class TypeOfPayload(Enum):
    Electronic = 0
    Visible = 1
    Infrared = 2
    SAR = 3


# payload_priorities = {
#     TypeOfPayload.Electronic.value: 0.8,
#     TypeOfPayload.Visible.value: 1,
#     TypeOfPayload.Infrared.value: 1,
#     TypeOfPayload.SAR.value: 0.6
# }


class Satellite(object):
    def __init__(self, id=None, name=None, start_datetime=None, scenario_id=None, half_angle=None, effectiveness_loss=None, additive_bias=None, max_omega=METADATA['max_omega']):

        self.id = id
        self.name = name
        self.scenario_id = scenario_id

        self.txt = None
        self.sampling_period = 1
        self.index = 0

        # inertia matrix
        self.I = np.array([
            [10, 0.02, 0.01],
            [0.02, 15, -0.01],
            [0.01, -0.01, 20]
        ])

        # fault information
        self.effectiveness_loss = effectiveness_loss
        self.additive_bias = additive_bias

        # historical trajectory
        self.historical_trajectory = []

        # for task allocation
        self.occultation_list = None
        self.profit_list = []

        # initial attitude
        self.r_ob = None

        self.omega = None  # the current angular velocity
        self.quaternion0 = None
        self.max_omega = max_omega

        # Field of View
        self.half_angle = half_angle

        # information (only for init, will change with reset())
        self.start_datetime = start_datetime
        self.datetime = start_datetime

        # position and velocity of satellite
        self.s_eci = None
        self.s_lla = None
        self.s_velocity_eci = None

        # for tracking record
        self.is_tracking = []
        self.angles = {}

        # orbital frame
        self.x0 = None
        self.y0 = None
        self.z0 = None

        # Keplerian orbital elements
        self.true_anomaly = None
        self.argp = None
        self.incl = None
        self.raan = None
        # rotation matrix from eci frame to orbit frame
        self.Reo = None

    def read_txt(self):
        txt_path = os.path.join(
            '../../envs/marltracking/tasks/task' + str(METADATA['scenario_id']), r'Satellite' + str(self.name).replace("Sat", "") + r' Information.txt')
        txt_file = open(txt_path, 'r')
        lines = txt_file.readlines()
        txt_file.close()
        return lines

    def eci2orbit(self, vector_eci):
        return self.Reo @ vector_eci

    def orbit2eci(self, vector_orbit):
        return inv(self.Reo) @ vector_orbit

    def body2orbit(self, vector_body):
        return inv(self.r_ob) @ vector_body

    def orbit2body(self, vector_orbit):
        return self.r_ob @ vector_orbit

    def reset(self, start_datetime):

        # reset the info
        self.txt = self.read_txt()

        # initial attitude
        self.quaternion0 = np.array([0, 0, 0, 1])
        self.omega = np.array([0, 0, 0])

        # initial actuator faults
        if METADATA['with_faults']:
            if METADATA['mode'] == 'train':
                self.random_faults_for_training()
        else:
            self.wo_faults()

        self.datetime = start_datetime
        self.index = 0
        self.historical_trajectory = []

        self.update_position(start_datetime)

        self.r_ob = quaternion2matrix(self.quaternion0)  # rotation matrix from body frame to orbit frame

        # reset the detect time and the detect omega
        self.under_detect_list = []
        self.detect_omega_list = []

    def random_faults_for_training(self):
        self.effectiveness_loss = np.random.uniform(0, 0.5, 3)
        self.additive_bias = np.random.uniform(-0.05, 0.05, 3)

    def wo_faults(self):
        self.effectiveness_loss = np.zeros(3)
        self.additive_bias = np.zeros(3)

    def update_position(self, datetime):
        datetime0 = datetime.strptime(METADATA['start_datetime'], "%d %b %Y %H:%M:%S.%f")

        self.index = int((datetime - datetime0).total_seconds() / self.sampling_period)

        information = [i.strip() for i in self.txt[7 + self.index].strip().split('    ') if i != '']

        assert str2datetime(information[0]) == datetime, [str2datetime(information[0]), datetime,
                                                          'Time ERROR']

        x_eci = float(information[1])
        y_eci = float(information[2])
        z_eci = float(information[3])

        x_velocity_eci = float(information[4])
        y_velocity_eci = float(information[5])
        z_velocity_eci = float(information[6])

        self.s_eci = np.array([x_eci, y_eci, z_eci])
        self.s_velocity_eci = np.array([x_velocity_eci, y_velocity_eci, z_velocity_eci])

        # Y. Lian, Y. Gao, and G. Zeng
        # Staring imaging attitude control of small satellites
        Rs = self.s_eci
        Vs = self.s_velocity_eci

        h_hat = cross(Rs, Vs) / norm(cross(Rs, Vs))
        self.z0 = -Rs / norm(Rs)
        self.y0 = - h_hat
        self.x0 = cross(self.y0, self.z0)

        self.true_anomaly = float(information[7].strip())
        self.argp = float(information[8])
        self.incl = float(information[9])
        self.raan = float(information[10])
        self.Reo = get_Reo(self.true_anomaly, self.argp, self.incl, self.raan)

    def update_attitude(self, command_torque):

        # the external disturbance
        disturbance = generate_external_disturbance(t=self.index, omega=self.omega, constant=False)

        # the total torque equals to the sum of control_torque and external_disturbance
        actual_tau = (np.ones_like(self.effectiveness_loss) - self.effectiveness_loss) * command_torque + self.additive_bias
        # tau = actual_tau + disturbance
        tau = actual_tau + disturbance

        # integrate
        new_omega = integrate_omega(self.omega, self.sampling_period, tau, self.I, max_omega_for_clip=METADATA['max_omega_for_clip'])
        new_quaternion = integrate_quaternion(self.quaternion0, self.sampling_period, new_omega)

        # update the current attitude
        self.omega = new_omega

        self.quaternion0 = new_quaternion
        self.r_ob = quaternion2matrix(self.quaternion0)

    def update(self, torques):

        self.datetime = next_dt(self.datetime, self.sampling_period)

        # update position, attitude and state (t -> t+1)
        self.update_position(self.datetime)
        self.update_attitude(torques)

        # if np.any(self.omega < -self.max_omega) or np.any(self.omega > self.max_omega):
        #     print("Out of the safety margin !!!")

    def get_desired_quaternion(self, t_eci):
        ###
        rho = t_eci - self.s_eci

        zb = rho / norm(rho)
        xb = cross(zb, -self.y0) / norm(cross(zb, -self.y0))
        yb = cross(zb, xb)

        # orbital to body
        R_desired = np.array([
            [dot(xb, self.x0), dot(xb, self.y0), dot(xb, self.z0)],
            [dot(yb, self.x0), dot(yb, self.y0), dot(yb, self.z0)],
            [dot(zb, self.x0), dot(zb, self.y0), dot(zb, self.z0)]])

        # Three desired angles
        roll_angle = degrees(arctan2(dot(yb, self.z0), dot(zb, self.z0)))
        pitch_angle = degrees(arcsin(dot(-xb, self.z0)))
        yaw_angle = 0

        ###

        # # for the effect of rotation sequence
        # st_orbit = self.eci2orbit(t_eci - self.s_eci)
        # st_orbit = st_orbit / norm(st_orbit)
        #
        # seq_123 = False
        # x, y, z = st_orbit
        # if seq_123:  # x-y-z
        #     roll_angle = np.degrees(-arctan2(y, z))
        #     pitch_angle = np.degrees(arcsin(x))
        #     yaw_angle = 0
        #     # print(roll_angle, pitch_angle, yaw_angle)
        #     R_desired = rotate_matrix_y(np.radians(pitch_angle)) @ rotate_matrix_x(np.radians(roll_angle))

        ###

        quaternion_desired = matrix2quaternion(R_desired)

        return R_desired, [roll_angle, pitch_angle, yaw_angle], quaternion_desired

    def get_desired_and_error_quaternion(self, t_eci):
        _, _, desired_quaternion = self.get_desired_quaternion(t_eci)
        conjugate_desired_quaternion = get_conjugate_quaternion(desired_quaternion)
        error_quaternion = quaternion_multiply(conjugate_desired_quaternion, self.quaternion0)

        # print(self.quaternion0, desired_quaternion)

        return desired_quaternion, error_quaternion

    def get_angles(self, t_eci):
        st_eci = t_eci - self.s_eci
        st_orbit = self.eci2orbit(st_eci)

        # [1, 0, 0] is the normal vector of the plane YOZ
        x_axis = self.body2orbit(np.array([1, 0, 0]))
        horizontal_half_angle = np.abs(90 - angle_between_vectors(st_orbit, x_axis))

        assert 0 <= horizontal_half_angle <= 90, 'horizontal half angle {} should between [0, 90]'.format(horizontal_half_angle)

        # [0, 1, 0] is the normal vector of the plane XOZ
        y_axis = self.body2orbit(np.array([0, 1, 0]))
        vertical_half_angle = np.abs(90 - angle_between_vectors(st_orbit, y_axis))

        assert 0 <= vertical_half_angle <= 90, 'vertical half angle {} should between [0, 90]'.format(vertical_half_angle)

        # [0, 1, 0] is the z-axis
        z_axis = self.body2orbit(np.array([0, 0, 1]))
        angle_ = angle_between_vectors(st_orbit, z_axis)

        return horizontal_half_angle, vertical_half_angle, angle_

    def is_occultated(self, t_eci):

        ts_eci = self.s_eci - t_eci
        alpha_angle = angle_between_vectors(ts_eci, t_eci)

        radius = 6357  # radius of earth
        height = norm(t_eci) - radius  # height of aircraft

        if alpha_angle <= 180 - np.degrees(arctan2(radius, radius + height)):
            return False

        return True

    def observation(self, targets):

        obs_list = []

        for tar in targets:
            desired_quaternion, error_quaternion = self.get_desired_and_error_quaternion(tar.t_eci if METADATA['mode'] == "train" else tar.t_eci_hat)
            obs_list = np.append(obs_list, error_quaternion)

        # return np.concatenate([obs_list, self.omega, self.effectiveness_loss, self.additive_bias]).astype(np.float32)
        return np.concatenate([obs_list, self.omega]).astype(np.float32)

    def satellite_reward(self, targets):

        # for tracking record
        self.is_tracking = []
        self.angles = {}

        maneuver_reward = 0
        stability_reward = 0

        # N_tar = len(targets)

        for tar in targets:
            _, error_quaternion = self.get_desired_and_error_quaternion(tar.t_eci)
            _, _, _, qe_4 = error_quaternion
            occulated = self.is_occultated(tar.t_eci)

            horizontal_half_angle, vertical_half_angle, angle_ = self.get_angles(tar.t_eci)

            self.angles[tar.name] = angle_

            # maneuver rewaed
            maneuver_reward += qe_4

            # for tracking reward (t_reward defined outer)
            is_visible = qe_4 > 0.8 and horizontal_half_angle <= self.half_angle[0] and vertical_half_angle <= self.half_angle[1] and (not occulated)
            if is_visible:
                tar.under_tracking.append(self.name)
                self.is_tracking.append(tar.name)

                # stability reward
                stability_reward = -np.sum(np.abs(self.omega))

        # safety margin reward
        safety_reward = 0
        if np.any(self.omega < -self.max_omega) or np.any(self.omega > self.max_omega):
            # print("Out of the safety margin !!!")
            safety_reward = -15

        sum_reward = 1 * maneuver_reward + 10 * stability_reward + 1 * safety_reward
        return sum_reward
