import numpy as np
import math
from numpy.linalg import norm
from numpy import (pi, sin, cos, tan, arcsin, arccos, arctan, arctan2, sqrt, abs)
from pprint import pprint
from datetime import datetime, timedelta
from scipy.optimize import curve_fit


def calc_roll_angle(st_vvlh):
    x, y, z = st_vvlh
    return np.degrees(- arctan2(y, z))


def calc_pitch_angle(st_vvlh):
    x, y, z = st_vvlh
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return np.degrees(arcsin(x / r))


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / norm_product

    # cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0) # avoid overflow

    angle = np.arccos(cos_theta)
    return np.degrees(angle)


def str2datetime(time_string):
    dt = datetime.strptime(time_string, "%d %b %Y %H:%M:%S.%f")
    return dt


def datetime2str(datetime):
    time_string = datetime.strftime("%d %b %Y %H:%M:%S.%f")
    return time_string[:-3]


def next_dt(datetime, delta=0.25):
    delta = timedelta(seconds=delta)
    next_dt = datetime + delta
    return next_dt


def get_azimuth_elevation(s, t):
    # s_eci = np.array([4,5,6])
    # t_eci = np.array([1,2,2])
    s_eci, t_eci = s, t

    ts_eci = s_eci - t_eci

    azimuth = np.arctan2(ts_eci[0], ts_eci[1])
    elevation = np.arcsin(ts_eci[2] / norm(ts_eci))

    return azimuth, elevation




if __name__ == '__main__':
    historical_trajectory = [
        [77.076, 26.917],
        [77.078, 26.917],
        [77.079, 26.918],
        [77.081, 26.918],
        [77.083, 26.919]
    ]

    get_trajectory_vector(historical_trajectory)
