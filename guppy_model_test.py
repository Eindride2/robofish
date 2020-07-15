import numpy as np
import torch
import torch.nn as nn
from guppy_model import *
from os.path import isfile, join
from os import listdir
from view_hdf import Guppy_Dataset
from torch.utils.data import Dataset, DataLoader
from hyper_params import *

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

predictions = []
loss = 0
with torch.no_grad():
    for input, targets in testloader:
            targets = targets.view(targets.shape[0] * targets.shape[1], 2)
            angle_targets = targets[:, 0]
            speed_targets = targets[:, 1]
            angle_pred, speed_pred,_ = model.forward(input)
            predictions.append(output)
            loss1 = loss_function(angle_pred, angle_targets)
            loss2 = loss_function(speed_pred, speed_targets)
            loss += loss1 + loss2

loss = loss / dataset.length
print(f'test loss: {loss.item():10.10f}')

