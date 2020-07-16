import numpy as np
import torch
import torch.nn as nn
from guppy_model import *
from os.path import isfile, join
from os import listdir
from view_hdf import Guppy_Dataset
from torch.utils.data import Dataset, DataLoader
from hyper_params import *
from auxiliary_funcs import get_prediction_bins

torch.set_default_dtype(torch.float64)

if output_model == "multi_modal":
    model = LSTM_multi_modal()
    loss_function = nn.CrossEntropyLoss()
else:
    model = LSTM_fixed()
    loss_function = nn.MSELoss()

model.load_state_dict(torch.load(network_path))
model.eval()

testpath = "guppy_data/live_female_female/test/" if live_data else "guppy_data/couzin_torus/test/"
files = [join(testpath, f) for f in listdir(testpath) if
         isfile(join(testpath, f)) and f.endswith(".hdf5")]
files.sort()
num_files = len(files)
files = files[:num_files]
print(files)

dataset = Guppy_Dataset(files, 0, num_guppy_bins, num_wall_rays, livedata=False,output_model = output_model)
testloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

#predictions = []

with torch.no_grad():

    loss = acc_turn = acc_speed = conf_turn = conf_speed = 0

    for input, targets in testloader:

        targets = targets.view(targets.shape[0] * targets.shape[1], 2)
        angle_targets = targets[:, 0]
        speed_targets = targets[:, 1]
        angle_pred, speed_pred,_ = model.forward(input)
        angle_pred = angle_pred.view(angle_pred.shape[0] * angle_pred.shape[1], -1)
        speed_pred = speed_pred.view(speed_pred.shape[0] * speed_pred.shape[1], -1)

        angle_bin_pred, speed_bin_pred, angle_prob_pred, speed_prob_pred = \
            get_prediction_bins(angle_pred, speed_pred)

        ##### scores #####

        if output_model == "multi_modal":
            conf_turn += np.mean(angle_prob_pred)
            conf_speed += np.mean(speed_prob_pred)

            marginal = 0
            max_angle_bin = angle_targets + marginal
            min_angle_bin = angle_targets - marginal
            max_speed_bin = speed_targets + marginal
            min_speed_bin = speed_targets - marginal

            for ind in range(len(angle_bin_pred)):

                if min_angle_bin[ind] <= angle_bin_pred[ind] <= max_angle_bin[ind]:
                    acc_turn += 1
                if min_speed_bin[ind] <= speed_bin_pred[ind] <= max_speed_bin[ind]:
                    acc_speed += 1

            acc_turn = acc_turn / len(angle_bin_pred)
            acc_speed = acc_speed / len(speed_bin_pred)

        loss1 = loss_function(angle_pred, angle_targets)
        loss2 = loss_function(speed_pred, speed_targets)
        loss += loss1

if output_model == "multi_modal":
     conf_turn = conf_turn * batch_size / dataset.length
     conf_speed = conf_speed * batch_size / dataset.length
     confidence_turn.append(conf_turn)
     confidence_speed.append(conf_speed)
     acc_turn = acc_turn * batch_size / dataset.length
     acc_speed = acc_speed * batch_size / dataset.length
     accuracy_turn.append(acc_turn)
     accuracy_speed.append(acc_speed)

     print(f'epoch: {i:3} average confidence turn: {conf_turn:10.10f}')
     print(f'epoch: {i:3} average confidence speed: {conf_speed:10.10f}')
     print(f'epoch: {i:3} accuracy turn: {acc_turn:10.10f}')
     print(f'epoch: {i:3} accuracy speed: {acc_speed:10.10f}')

loss = loss / dataset.length
print(f'test loss: {loss.item():10.10f}')

