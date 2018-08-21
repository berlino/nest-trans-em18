import itertools

import numpy as np
import torch.nn as nn
import torch.nn.init


def xavier_init(if_gpu, *size):
    var = nn.init.xavier_normal_(torch.FloatTensor(*size))
    if if_gpu: var = var.cuda()
    return nn.Parameter(var)


def init_embedding(input_embedding):
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)


def init_linear(input_linear):
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_lstm_cell(input_lstm):
    weight = eval('input_lstm.weight_ih')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)
    weight = eval('input_lstm.weight_hh')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        weight = eval('input_lstm.bias_ih' )
        weight.data.zero_()
        weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        weight = eval('input_lstm.bias_hh')
        weight.data.zero_()
        weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

