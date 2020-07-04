import numpy as np
import torch
import torch.nn as nn
from view_hdf import vec_to_angle
from hyper_params import *
from view_hdf import Guppy_Calculator, Guppy_Dataset
from torch.utils.data import Dataset, DataLoader

loss_function = nn.MSELoss()


# inspired by https://github.com/LeanManager/NLP-PyTorch/blob/master/Character-Level%20LSTM%20with%20PyTorch.ipynb
# TODO: Klassifizierung vs Regression
# TODO: Prediction, Training Error!
class LSTM_fixed(nn.Module):
    def __init__(self, input_size=input_dim, hidden_layer_size=hidden_layer_size):
        # output size has to be the number of bins for first loc vec component + for the second
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)


        # predict the two components
        self.linear = nn.Linear(hidden_layer_size, 2)
        self.hidden_state = self.init_hidden(batch_size, num_layers)

    def forward(self, x, hc):
        # print("Input seq: ", input_seq.view(1,1,len(input_seq)))
        # print("Hidden Cell: ", self.hidden_cell)

        x, (h, c) = self.lstm(x, hc)

        out = self.linear(x)

        return out, (h, c)

    # return angle_out, speed_out, (h, c)

    def predict(self, x, y):
        timesteps = np.shape(x)[1]
        x_dim = np.shape(x)[2]
        y_dim = np.shape(y)[2]
        this_data = x[:, 0, :]
        this_data = torch.reshape(this_data, (-1, 1, x_dim))
        pred =[]

        for i in range(timesteps):

            x, (h, c) = self.lstm(this_data, self.hidden_state)
            pred_pos = self.linear(x)
            pred.append(pred_pos)

            this_pos = this_data[:,:,:2] # exclude sensory data

            input = torch.cat((pred_pos, this_pos), axis = 1) #size: (batch_size,2,2)

            data = Guppy_Dataset(None, agent, input, num_guppy_bins, num_wall_rays, live_data, output_model)
            this_data = DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=False)

        loss = loss_function(pred, y)
        print(loss.item())
        return pred, loss


    def init_hidden(self, batch_size, num_layers):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, self.hidden_layer_size).zero_(),
                weight.new(num_layers, batch_size, self.hidden_layer_size).zero_())

    def simulate(self, initial_pose, initial_loc_sensory, frames):
        pos = initial_pose[0], initial_pose[1]
        ori = vec_to_angle(initial_pose[2], initial_pose[3])

        # for i in range(frames):


class LSTM_multi_modal(nn.Module):
    def __init__(self, input_size=input_dim, hidden_layer_size=hidden_layer_size):
        # output size has to be the number of bins for first loc vec component + for the second
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        self.linear1 = nn.Linear(hidden_layer_size, num_angle_bins)
        self.linear2 = nn.Linear(hidden_layer_size, num_speed_bins)

        # predict the two components
        self.hidden_state = self.init_hidden(batch_size, num_layers)

    def forward(self, x, hc):
        # print("Input seq: ", input_seq.view(1,1,len(input_seq)))
        # print("Hidden Cell: ", self.hidden_cell)

        x, (h, c) = self.lstm(x, hc)

        angle_out = self.linear1(x)
        speed_out = self.linear2(x)

        return angle_out, speed_out, (h, c)


    def predict(self, test_ex, label):
        # not ready
        x, (h, c) = self.lstm(test_ex, self.hidden_state)
        out = self.linear(x)
        loss = loss_function(out, label)
        print(loss.item())

    def init_hidden(self, batch_size, num_layers):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, self.hidden_layer_size).zero_(),
                weight.new(num_layers, batch_size, self.hidden_layer_size).zero_())

    def simulate(self, initial_pose, initial_loc_sensory, frames):
        pos = initial_pose[0], initial_pose[1]
        ori = vec_to_angle(initial_pose[2], initial_pose[3])

        # for i in range(frames):
