import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from view_hdf import get_locomotion_vec, Guppy_Calculator, Guppy_Dataset
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from guppy_model import LSTM_fixed, LSTM_multi_modal
import sys
import copy
from hyper_params import *
from evaluate_performance import plot_scores
from auxiliary_funcs import get_prediction_bins

torch.manual_seed(1)

# get the files for 4, 6 and 8 guppys
trainpath = "guppy_data/live_female_female/train/" if live_data else "guppy_data/couzin_torus/train/"
files = [join(trainpath, f) for f in listdir(trainpath) if isfile(join(trainpath, f)) and f.endswith(".hdf5") ]
files.sort()
num_files = len(files) // 2
files = files[-30:]
print(files)

torch.set_default_dtype(torch.float64)

# now we use a regression model, just predict the absolute values of linear speed and angular turn
# so we need squared_error loss

if output_model == "multi_modal":
    model = LSTM_multi_modal()
    loss_function = nn.CrossEntropyLoss()
else:
    model = LSTM_fixed()
    loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
print(model)
# training

dataset = Guppy_Dataset(files, 0, num_guppy_bins, num_wall_rays, livedata=live_data, output_model=output_model, max_agents= 1)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

train_losses = []
val_losses = []
# confidences = []
confidence_turn = []
confidence_speed = []
accuracy_turn = []
accuracy_speed = []

epochs = 30
seq_len = 20
for i in range(epochs):
    try:
        loss_score = 0  # only for storing loss, not for updating
        val_loss = 0
        acc_turn = 0
        acc_speed = 0
        conf_turn = conf_speed = 0

        #states = [model.init_hidden(batch_size, 1, hidden_layer_size) for _ in range(num_layers * 2)]
        #h = model.init_hidden(batch_size, num_layers, hidden_layer_size)
        #loss = 0
        for inputs, targets in dataloader:
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            optimizer.zero_grad()
            #states = [tuple([each.data for each in s]) for s in states]
            states = [model.init_hidden(batch_size, 1, hidden_layer_size) for _ in range(num_layers * 2)] if arch == "ey" \
                else model.init_hidden(batch_size, num_layers, hidden_layer_size)

            if output_model == "multi_modal":
                targets = targets.type(torch.LongTensor)
                #loss = 0
                count = 0
                for s in range(0, inputs.size()[1] - seq_len, seq_len):
                    count +=1
                    states = [tuple([each.data for each in s]) for s in states] if arch == "ey" else \
                        tuple([each.data for each in states])
                    angle_pred, speed_pred, states = model.forward(inputs[:, s:s + seq_len, :], states)

                    #angle_pred, speed_pred, states = model.forward(inputs[:, s:s + seq_len, :], states)
                    angle_pred = angle_pred.view(angle_pred.shape[0] * angle_pred.shape[1], -1)
                    speed_pred = speed_pred.view(speed_pred.shape[0] * speed_pred.shape[1], -1)
                    seq_targets = targets[:, s: s + seq_len, :]
                    seq_targets = seq_targets.contiguous().view(seq_targets.shape[0] * seq_targets.shape[1], -1)
                    angle_targets = seq_targets[:, 0]
                    speed_targets = seq_targets[:, 1]

                    # scores

                    angle_bin_pred, speed_bin_pred, angle_prob_pred, speed_prob_pred = \
                        get_prediction_bins(angle_pred, speed_pred)

                    #print(len(angle_prob_pred))

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
                    loss_score += loss1 + loss2

                    loss = loss1 + loss2
                    loss.backward()
                    optimizer.step()

                    torch.set_printoptions(threshold=10000)
                    with torch.no_grad():
                        for j in range(1):
                            angle_probs = nn.Softmax(0)(angle_pred[j])
                            speed_probs = nn.Softmax(0)(speed_pred[j])
                            #print("angle prob:\n", angle_probs[angle_targets[j].data])
                            #print(angle_targets[i])
                #loss /= inputs.shape[1] // seq_len

            else:
                #loss = 0
                for s in range(0, inputs.size()[1], seq_len):
                    #states = [tuple([each.data for each in s]) for s in states] if arch == "ey" else \
                    #    tuple([each.data for each in states])
                    prediction, states = model.forward(inputs[:, s:s + seq_len, :], states)
                    loss = loss_function(prediction, targets[:, s: s + seq_len, :])

                #loss /= inputs.shape[1] // seq_len
                loss.backward()
                optimizer.step()

        if output_model == "multi_modal":
            conf_turn = conf_turn *  batch_size /(count * dataset.length)
            conf_speed = conf_speed * batch_size /(count * dataset.length)
            confidence_turn.append(conf_turn)
            confidence_speed.append(conf_speed)
            acc_turn = acc_turn * batch_size /(count * dataset.length)
            acc_speed = acc_speed * batch_size /(count * dataset.length)
            accuracy_turn.  append(acc_turn)
            accuracy_speed.append(acc_speed)

            print(f'epoch: {i:3} average confidence turn: {conf_turn:10.10f}')
            print(f'epoch: {i:3} average confidence speed: {conf_speed:10.10f}')
            print(f'epoch: {i:3} accuracy turn: {acc_turn:10.10f}')
            print(f'epoch: {i:3} accuracy speed: {acc_speed:10.10f}')

        loss_score = loss_score / (count * dataset.length)
        train_losses.append(loss_score)

        print("###################################")
        print(f'epoch: {i:3} loss: {loss.item():10.10f}')
        print("###################################")
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
       #loss.backward()
       # optimizer.step()

    except KeyboardInterrupt:
        if input("Do you want to save the model trained so far? y/n") == "y":
            torch.save(model.state_dict(), network_path + f".epochs{i}")
            print("network saved at " + network_path + f".epochs{i}")
        sys.exit(0)



torch.save(model.state_dict(), network_path + f".epochs{epochs}")
print("network saved at " + network_path + f".epochs{epochs}")

