import torch.nn as nn
import torch
from torch.autograd import Variable
import argparse
import numpy as np
import torch.nn.functional as F
from util.graph_definition import *

class conv_lstm(nn.Module):
    def __init__(self, hidden_size, kernel, stride, nb_filter, input_size):
        super(conv_lstm, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(input_size, nb_filter, kernel, stride),
                                nn.ReLU(),
                                nn.BatchNorm1d(nb_filter)
        )
        self.lstm = create_model(model='skip_lstm',
                             input_size=nb_filter,
                             hidden_size=hidden_size,
                             num_layers=1)
        #self.lstm = lstm_cell(input_size=nb_filter, hidden_size=hidden_size, batch_first=True, layer_norm=True)
        self.hidden_size = hidden_size

    def forward(self, input):
        input = self.conv(input.permute(0, 2, 1))
        input = input.permute(0, 2, 1)
        output = self.lstm(input)
        output, hx, updated_state = split_rnn_outputs('skip_lstm', output)
        return output[:, -1, :]

class Scoring(nn.Module):

    def __init__(self, hidden_size, feature_size):
        super(Scoring, self).__init__()

        self.lstm_direction = 1
        self.conv = nn.Sequential(
                        nn.Conv1d(feature_size, 512, 1, 1),
                        nn.ReLU(),
                        nn.BatchNorm1d(512)
        )

        self.scale1 = conv_lstm(hidden_size, 2, 1, 256, 512)
        self.scale2 = conv_lstm(hidden_size, 4, 2, 256, 512)
        self.scale3 = conv_lstm(hidden_size, 8, 4, 256, 512)
        self.linear1 = nn.Linear(hidden_size*self.lstm_direction*1, 256)
        self.cls = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, model_input):
        model_input = model_input.permute(0, 2, 1)
        model_input = self.conv(model_input).permute(0, 2, 1)
        #output = torch.cat([self.scale1(model_input), self.scale2(model_input), self.scale3(model_input)], 1)
        output = self.scale1(model_input) + self.scale2(model_input) + self.scale3(model_input)
        output = self.relu(self.linear1(output))
        return self.cls(output)

    def loss(self, regression, actuals):
        """
        use mean square error for regression and cross entropy for classification
        """
        regr_loss_fn = nn.L1Loss()
        return regr_loss_fn(regression, actuals)

