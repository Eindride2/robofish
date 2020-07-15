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
from evaluate_performance import plot_scores
import sys
import copy
from hyper_params import *
import pickle
from auxiliary_funcs import get_prediction_bins

torch.manual_seed(1)

# get the files for 4, 6 and 8 guppys
trainpath = "guppy_data/live_female_female/train/" if live_data else "guppy_data/couzin_torus/train/"
testpath = "guppy_data/live_female_female/test/" if live_data else "guppy_data/couzin_torus/test/"
files = [join(trainpath, f) for f in listdir(trainpath) if isfile(join(trainpath, f)) and f.endswith(".hdf5") ]
test_files = [join(testpath, f) for f in listdir(testpath) if isfile(join(testpath, f)) and f.endswith(".hdf5") ]
files.sort()
test_files.sort
num_files = len(files)
files = files[97:] #all files with > 1 fish
test_files = test_files[10:] #ditto

files = files[0:1]
test_files = test_files[0:1]

torch.set_default_dtype(torch.float64)

# now we use a regression model, just predict the absolute values of linear speed and angular turn
# so we need squared_error loss

if output_model == "multi_modal":
    model = LSTM_multi_modal()
    loss_function = nn.CrossEntropyLoss()
else:
    model = LSTM_fixed()
    loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
# training

dataset = Guppy_Dataset(files, 0, num_guppy_bins, num_wall_rays, livedata=live_data, output_model=output_model)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
testdata = Guppy_Dataset(test_files, 0, num_guppy_bins, num_wall_rays, livedata=live_data, output_model=output_model)
testloader = DataLoader(testdata, batch_size=batch_size, drop_last=True, shuffle=True)

train_losses = []
val_losses = []
#confidences = []
confidence_turn = []
confidence_speed = []
accuracy_turn = []
accuracy_speed = []

for i in range(epochs):

    try:
        h = model.init_hidden(batch_size, num_layers, hidden_layer_size)
        states = [model.init_hidden(batch_size, 1, hidden_layer_size) for _ in range(num_layers * 2)]
        loss = 0
        val_loss = 0
        acc_turn = 0
        acc_speed = 0
        conf_turn = conf_speed = 0

        for inputs, targets in dataloader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            model.zero_grad()
            #h = tuple([each.data for each in h])
            states = [tuple([each.data for each in s]) for s in states]

            if output_model == "multi_modal":
                targets = targets.type(torch.LongTensor)

               # angle_pred, speed_pred, h = model.forward(inputs, h)
                angle_pred, speed_pred, states = model.forward(inputs, states)

                #print(angle_bins)
                #print(angle_pred.size())
                #print(speed_pred.size())

                #print(targets.size())
                angle_pred = angle_pred.view(angle_pred.shape[0] * angle_pred.shape[1], -1)
                speed_pred = speed_pred.view(speed_pred.shape[0] * speed_pred.shape[1], -1)
                angle_bin_pred, speed_bin_pred, angle_prob_pred, speed_prob_pred = \
                    get_prediction_bins(angle_pred, speed_pred)

                #print(speed_pred.size())
                targets = targets.view(targets.shape[0] * targets.shape[1], 2)
                #print(targets.size())
                angle_targets = targets[:, 0]
                speed_targets = targets[:, 1]

                conf_turn += np.mean(angle_prob_pred)
                conf_speed += np.mean(speed_prob_pred)

                # accuracy - proportion of predictions falling in correct bins

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

                with torch.no_grad():
                    for j in range(100):
                        angle_probs = nn.Softmax(0)(angle_pred[i])
                        print(angle_probs[angle_targets[i].data])
                        print(angle_targets[i])

                loss1 = loss_function(angle_pred, angle_targets)
                loss2 = loss_function(speed_pred, speed_targets)
                loss += loss1 + loss2

                # print("------ANGLE SCORES-------")
                # print(angle_pred)
                # print("------ANGLE TARGETS -------")
                # print(angle_targets)
                # with torch.no_grad():
                # print("------SPEED PROBS-------")
                # with torch.no_grad():
                #     print(nn.Softmax(0)(speed_pred[0]))
                #     print("------SPEED TARGETS -------")
                #     print(speed_targets)

            else:
                prediction, h = model.forward(inputs, h)
                loss += loss_function(prediction, targets)

    except KeyboardInterrupt:
            if input("Do you want to save the model trained so far? y/n") == "y":
                torch.save(model.state_dict(), network_path + f".epochs{i}")
            sys.exit(0)

    timesteps = len(angle_pred) / batch_size

    if output_model == "multi_modal":

        conf_turn = conf_turn * (batch_size / dataset.length)
        conf_speed = conf_speed * (batch_size / dataset.length)
        confidence_turn.append(conf_turn)
        confidence_speed.append(conf_speed)
        acc_turn = acc_turn * (batch_size / dataset.length)
        acc_speed = acc_speed * (batch_size / dataset.length)
        accuracy_turn.append(acc_turn)
        accuracy_speed.append(acc_speed)

        print(f'epoch: {i:3} average confidence turn: {conf_turn:10.10f}')
        print(f'epoch: {i:3} average confidence speed: {conf_speed:10.10f}')
        print(f'epoch: {i:3} accuracy turn: {acc_turn:10.10f}')
        print(f'epoch: {i:3} accuracy speed: {acc_speed:10.10f}')

    loss = loss / dataset.length
    train_losses.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()
    print(f'epoch: {i:3} training loss: {loss.item():10.10f}')


########validation#######

    model.eval()
    for inputs, targets in testloader:

        if output_model == "multi_modal":

            targets = targets.type(torch.LongTensor)
            angle_pred, speed_pred,_ = model.forward(inputs,states)

            angle_pred = angle_pred.view(angle_pred.shape[0] * angle_pred.shape[1], -1)
            speed_pred = speed_pred.view(speed_pred.shape[0] * speed_pred.shape[1], -1)
            targets = targets.view(targets.shape[0] * targets.shape[1], 2)
            angle_targets = targets[:, 0]
            speed_targets = targets[:, 1]

            loss1 = loss_function(angle_pred, angle_targets)
            loss2 = loss_function(speed_pred, speed_targets)
            val_loss += loss1 + loss2

        else:
            prediction,_ = model.forward(inputs, h)
            val_loss += loss_function(prediction, targets)

    val_loss = val_loss / testdata.length
    val_losses.append(val_loss.detach().numpy())
    print(f'epoch: {i:3} validation loss: {val_loss:10.10f}')
    #torch.save(model.state_dict(), network_path + f".epochs{i}")

scores = [train_losses, val_losses, confidence_turn, confidence_speed, accuracy_turn, accuracy_speed]

torch.save(model.state_dict(), network_path)
print("network saved at " + network_path + f".epochs{epochs}")
with open('scores', 'wb') as f:
    pickle.dump(scores, f)

plot_scores(scores, load_from_file = False, filename = None)







