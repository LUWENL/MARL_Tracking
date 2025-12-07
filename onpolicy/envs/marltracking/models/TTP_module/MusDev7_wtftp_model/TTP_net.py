import argparse
import os
import random
import numpy as np
import math
import torch
from astropy.wcs.wcsapi.conftest import time_1d_fitswcs
from torch.utils.data import DataLoader
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from onpolicy.envs.marltracking.models.TTP_module.MusDev7_wtftp_model.model import WTFTPv2
# import sys
# sys.path.append("..")
from ..MusDev7_wtftp_model.pytorch_wavelets import DWT1DForward, DWT1DInverse
# from ..MusDev7_wtftp_model.pytorch_wavelets import DWT1DForward, DWT1DInverse
# from ..MusDev7_wtftp_model.model import WTFTP, WTFTP_AttnRemoved
from astropy.utils import iers

iers.conf.auto_download = False
iers.conf.auto_max_age = None
from astropy.coordinates import GCRS, ITRS, EarthLocation, CartesianRepresentation
from astropy import units as u
from astropy.time import Time

attr_names = ['lon', 'lat', 'alt']

data_ranges = {"lon": {"max": 179.998, "min": 30.037},
               "lat": {"max": 48.968, "min": -15.002},
               "alt": {"max": 20.039, "min": 11.949}}


def eci2lla(eci, dt):
    x, y, z = eci
    tt = Time(dt, format='datetime')

    gcrs = GCRS(CartesianRepresentation(x=x * u.km, y=y * u.km, z=z * u.km), obstime=tt)

    itrs = gcrs.transform_to(ITRS(obstime=tt))

    el = EarthLocation.from_geocentric(itrs.x, itrs.y, itrs.z)

    lon, lat, alt = el.to_geodetic('WGS84')

    return np.array([lon.value, lat.value, alt.value])


def lla2eci(lla, dt):
    lon, lat, alt = lla
    earth_location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg, height=alt * u.km)

    tt = Time(dt, format='datetime')

    gcrs = earth_location.get_gcrs(obstime=tt)
    eci = gcrs.represent_as(CartesianRepresentation)

    return np.array([eci.x.value, eci.y.value, eci.z.value])


def scale(inputs):
    '''
    :param inputs: [lon, lat, alt]
    :return: normalized [lon, lat, alt]
    '''

    outputs = []

    for i in range(len(inputs)):
        input = inputs[i]
        attr = attr_names[i]

        output = (input - data_ranges[attr]['min']) / (data_ranges[attr]['max'] - data_ranges[attr]['min'])

        outputs.append(output)

    return np.array(outputs)


def unscale(inputs):
    '''
    :param inputs: normalized [lon, lat, alt]
    :return:  [lon, lat, alt]
    '''
    outputs = []

    for i in range(len(inputs)):
        input = inputs[i]
        attr = attr_names[i]

        output = input * (data_ranges[attr]['max'] - data_ranges[attr]['min']) + data_ranges[attr]['min']

        outputs.append(output)

    return np.array(outputs)


def str2datetime(time_string):
    dt = datetime.strptime(time_string, "%d %b %Y %H:%M:%S.%f")
    return dt


class TTP_net():
    def __init__(self, cpu=False, model_path=r'../MusDev7_wtftp_model/pretrained_models/PWTFTP.pt'):
        self.cpu = cpu
        self.model_path = model_path

        self.net = WTFTPv2(n_inp=3, n_oup=3, his_step=5, en_layers=4)
        self.idwt = None

        self.iscuda = torch.cuda.is_available()
        # self.device = f'cuda:{torch.cuda.current_device()}' if self.iscuda and not self.cpu else 'cpu'
        self.device = 'cpu'

    def load_model(self):
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=False))

        # self.idwt = DWT1DInverse(wave=self.net.args['train_opt'].wavelet, mode=self.net.args['train_opt'].wt_mode).to(
        #     self.device)
        self.idwt = DWT1DInverse(wave="haar", mode="symmetric").to(self.device)

        self.net.eval()

    def trajectory_predict(self, input_array):
        '''
        predict the k+1 th position of target based on the previous k trajectory
        scaled lla -> unscaled lla or eci

        :param input_array: scaled lla np.array shape [8, 3]
        :param time: datetime
        :param s_measured_list: If false, mask the data measured by satellite
        :param convert2eci: convert2eci lla to eci ?
        :return: np.array shape [1, 3] unscaled lla if not convert2eci else eci
        '''

        input_array = np.array(input_array)

        input_tensor = torch.tensor(input_array.reshape(1, 5, 3)).to(torch.float32).to(self.device)

        # infer
        wt_pre_batch, _ = self.net(input_tensor)
        pre_batch = self.idwt((wt_pre_batch[-1].transpose(1, 2).contiguous(),
                               [comp.transpose(1, 2).contiguous() for comp in
                                wt_pre_batch[:-1]])).contiguous()
        pre_batch = pre_batch.transpose(1, 2)  # shape: batch * n_sequence * n_attr
        # print(pre_batch)
        output_tensor = pre_batch[:, 5, :]

        # print(input_tensor)
        # print('\n', pre_batch[:, 8, :])

        output_array = output_tensor.cpu().detach().numpy().reshape(3)

        # print(output_array)

        # unscale
        lla_hat = unscale(output_array)


        return lla_hat
