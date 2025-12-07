import numpy as np
import pandas as pd
import os
from .model_utils import str2datetime, datetime2str, next_dt
from ..metadata import METADATA
from datetime import datetime
from .TTP_module.MusDev7_wtftp_model.TTP_net import scale, lla2eci


class Target(object):
    def __init__(self, id=None, name = None, start_datetime=None, time_length=None, target_type='Aircraft', scenario_id=None, TTP_net = None):

        self.id = id
        self.name = name
        self.scenario_id = scenario_id

        self.target_type = target_type

        self.txt = None
        self.index_offset = 0
        self.sampling_period = 1  # 1s

        # information (only for init, will change with reset())
        self.start_datetime = start_datetime
        self.time_length = time_length
        self.datetime = self.start_datetime
        self.under_tracking = []

        self.t_eci = None
        self.t_lla = None

        # for trajecroty prediction
        self.ttp_net = TTP_net
        self.historical_trajectory = []
        self.t_eci_hat = self.t_eci

        # self.t_velocity = None
        self.in_region_list = [False for i in range(self.time_length)]

    def read_txt(self):

        txt_path = os.path.join('../../envs/marltracking/tasks/task' + str(METADATA["scenario_id"]), self.target_type + str(self.name).replace("Tar", "") + r' Information.txt')

        txt_file = open(txt_path, 'r')
        lines = txt_file.readlines()
        txt_file.close()

        datetime0 = datetime.strptime(METADATA['start_datetime'], "%d %b %Y %H:%M:%S.%f")
        datetime_ = datetime.strptime([i.strip() for i in lines[7].strip().split('    ') if i != ''][0],
                                      "%d %b %Y %H:%M:%S.%f")
        self.index_offset = int((datetime_ - datetime0).total_seconds() / self.sampling_period)
        return lines

    def update_position(self, datetime_):
        datetime0 = datetime.strptime(METADATA['start_datetime'], "%d %b %Y %H:%M:%S.%f")

        index = int((datetime_ - datetime0).total_seconds() / self.sampling_period) - self.index_offset

        if index < 0:
            return

        information = [i.strip() for i in self.txt[7 + index].strip().split('    ') if i != '']

        assert str2datetime(information[0]) == datetime_, [information[0], datetime2str(datetime_), 'Time ERROR']
        x_eci = float(information[1])
        y_eci = float(information[2])
        z_eci = float(information[3])

        # vx_eci = float(information[4])
        # vy_eci = float(information[5])
        # vz_eci = float(information[6])

        lon = float(information[4])
        lat = float(information[5])
        alt = float(information[6])

        self.t_eci = np.array([x_eci, y_eci, z_eci])
        self.t_lla = np.array([lon, lat, alt])
        # self.t_velocity = np.array([vx_eci, vy_eci, vz_eci])

        if METADATA['mode'] == 'train' or not METADATA['with_trajectory_prediction']:
            self.t_eci_hat = self.t_eci
        else:
            # for trajectory prediction
            self.historical_trajectory.append([lon, lat, alt])
            if len(self.historical_trajectory) >= 5:
                inputs = self.historical_trajectory[-5:]

                for i in range(5):
                    inputs[i] = scale(inputs[i])

                lla_hat = self.ttp_net.trajectory_predict(inputs)
                self.t_eci_hat = lla2eci(lla_hat, datetime_)
                # print(self.t_eci, self.t_eci_hat)


            else:
                self.t_eci_hat = self.t_eci


    def reset(self, start_datetime):
        # reset the id
        self.txt = self.read_txt()
        self.historical_trajectory = []
        self.datetime = start_datetime
        self.start_datetime = start_datetime  # record it to calculate delta_time
        self.under_tracking = []

        self.update_position(start_datetime)



    def update(self):
        self.datetime = next_dt(self.datetime, self.sampling_period)
        # update position and state (t -> t+1)
        self.update_position(self.datetime)
        self.under_tracking = []


if __name__ == '__main__':
    txt_path = os.path.join(r'D:\My Program\Pycharm Projects\MultiSatelliteTT\target_track_gym\targets\aircraft'
                            r'\Aircraft1 Information.txt')
    t1 = Target(1)
    t1.reset(datetime.strptime(r'1 Oct 2023 07:02:00.000', "%d %b %Y %H:%M:%S.%f"))
    print(t1.t_eci)
