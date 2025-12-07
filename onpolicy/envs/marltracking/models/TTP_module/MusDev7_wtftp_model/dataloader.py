import json
import logging
import os
import random
# import coordinate_conversion as cc
import numpy as np
import torch
import torch.utils.data as tu_data
import sys
from itertools import product
sys.path.append('../../..')





class DataGenerator:
    def __init__(self, data_path, minibatch_len, interval=1, train=True, test=True, train_shuffle=True,
                 test_shuffle=False, ):
        assert os.path.exists(data_path)
        self.attr_names = ['lon', 'lat', 'alt']
        self.data_path = data_path
        self.interval = interval
        self.minibatch_len = minibatch_len

        self.s_measured_possibilities = list(product([True, False], repeat=3))
        self.data_ranges = {"lon": {"max": 179.998, "min": 30.037},
                            "lat": {"max": 48.968, "min": -15.002},
                            "alt": {"max": 20.039 , "min": 11.949}}

        if train:
            self.train_set = mini_DataGenerator(
                self.readtxt(os.path.join(self.data_path, 'train_data'), shuffle=train_shuffle))
        if test:
            self.test_set = mini_DataGenerator(
                self.readtxt(os.path.join(self.data_path, 'test_data'), shuffle=test_shuffle))

        print('data range:', self.data_ranges)

    def readtxt(self, data_path, shuffle=True):
        print(data_path)
        assert os.path.exists(data_path)

        data = []
        for root, dirs, file_names in os.walk(data_path):
            for file_name in file_names:
                if not (file_name.endswith('txt') and file_name.startswith('Aircraft')):
                    continue
                with open(os.path.join(root, file_name)) as file:
                    lines = file.readlines()[7:]
                    lines = lines[::self.interval]
                    if len(lines) == self.minibatch_len:
                        data.append(lines)
                    elif len(lines) < self.minibatch_len:
                        continue
                    else:
                        for i in range(
                                len(lines) - self.minibatch_len + 1):  # i in [0, len(lines) - self.minibatch_len + 1]
                            data.append(lines[i:i + self.minibatch_len])
        print(f'{len(data)} items loaded from \'{data_path}\'')
        if shuffle:
            random.shuffle(data)
        return data

    def scale(self, inp, attr):
        assert type(attr) is str and attr in self.attr_names
        data_status = self.data_ranges
        inp = (inp - self.data_ranges[attr]['min']) / (self.data_ranges[attr]['max'] - self.data_ranges[attr]['min'])
        return inp

    def unscale(self, inp, attr):
        assert type(attr) is str and attr in self.attr_names
        data_status = self.data_ranges
        inp = inp * (self.data_ranges[attr]['max'] - self.data_ranges[attr]['min']) + self.data_ranges[attr]['min']
        return inp

    def collate(self, inp):
        '''
        :param inp: batch * n_sequence * n_attr
        :return:
        '''
        oup = []
        for minibatch in inp:
            tmp = []
            for line in minibatch:
                lon, lat, alt = line.strip().split(' ')
                tmp.append([float(lon), float(lat), float(alt)])

            # # mask
            # tmp = np.array(tmp)
            # tmp_mean = np.mean(tmp, axis=0)
            #
            # s_measured_list = random.choice(self.s_measured_possibilities)
            # for i in range(len(s_measured_list)):
            #     if not s_measured_list[i]:
            #         tmp[5 + i + 1] = tmp_mean

            minibatch = np.array(tmp)

            for i in range(minibatch.shape[-1]):
                minibatch[:, i] = self.scale(minibatch[:, i], self.attr_names[i])
            oup.append(minibatch)
            # print(np.array(oup).shape)
        return np.array(oup)


class mini_DataGenerator(tu_data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data_path = './transformed_data/' + 'train_data'
    for root, dirs, file_names in os.walk(data_path):
        for file_name in file_names:
            print(file_name)
            with open(os.path.join(root, file_name)) as file:
                lines = file.readlines()[7:]
                line = lines[32]
                information = line.strip().split(' ')
                print(information, type(information))
                lon, lat, alt = information
