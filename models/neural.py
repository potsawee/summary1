import torch
import torch.nn as nn
import numpy as np
import pdb

class AdaptiveGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, batch_first=True, dropout=0.1):
        super(AdaptiveGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        # parameters
        self.grucell = nn.GRUCell(input_size, hidden_size, bias)
        self.binary_gate = BinaryGate(input_size, hidden_size, hidden_size)

    def forward(self, input, h0, input_lengths):
        batch_size = input.size(0)
        num_utterances = input.size(1)

        output = torch.zeros((batch_size, num_utterances, self.hidden_size)).cuda()
        segment_indices = [[] for _ in range(batch_size)]

        for bn in range(batch_size):
            ht = h0
            for t in range(input_lengths[bn]):
                # GRU cell
                xt = input[bn:bn+1, t]
                ht = self.grucell(xt, ht)
                output[bn, t] = ht[0]

                # binary gate
                if t < input_lengths[bn]-1:
                    xt1 = input[bn:bn+1, t+1]
                    gt = self.binary_gate(xt1, ht)
                else:
                    gt = 1.0

                if gt > 0.5: # segmetation & reset GRU cell
                    segment_indices[bn].append(t)
                    ht = h0
                else: # no segmentation & pass GRU state
                    pass

        return output, segment_indices

class BinaryGate(nn.Module):
    def __init__(self, dim1, dim2, hidden_size):
        super(BinaryGate, self).__init__()
        self.linear1 = nn.Linear(dim1, hidden_size, bias=True)
        self.linear2 = nn.Linear(dim2, hidden_size, bias=True)
        self.linearF = nn.Linear(hidden_size, 1,    bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        """computes sigmoid( w^T(W1x1 + W2x2 + b) )
            x1.shape[-1] = dim1
            x2.shape[-1] = dim2
        """
        y = self.linear1(x1) + self.linear2(x2)
        z = self.sigmoid(self.linearF(y))

        return z
