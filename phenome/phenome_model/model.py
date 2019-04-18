from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from warpctc_pytorch import CTCLoss
from ctcdecode import CTCBeamDecoder
from phoneme_list import PHONEME_MAP
from weight_drop import WeightDrop

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable

import Levenshtein as L


def sort_pad_pack_batch(batch):
    lengths = np.asarray([datapoint.shape[0] for datapoint in batch])
    sort_order = np.argsort(-lengths)
    unsort_order = np.argsort(sort_order)
    sorted = batch[sort_order]
    padded = pad_sequence([torch.from_numpy(dp) for dp in sorted], batch_first=True)
    packed = pack_padded_sequence(padded, lengths[sort_order], batch_first=True)

    return packed.cuda(), sort_order, unsort_order, lengths

def sort_pad_batch(batch):
    lengths = np.asarray([datapoint.shape[0] for datapoint in batch])
    sort_order = np.argsort(-lengths)
    unsort_order = np.argsort(sort_order)
    sorted = batch[sort_order]
    padded = pad_sequence([torch.from_numpy(dp) for dp in sorted], batch_first=True)

    return padded.cuda(), sort_order, unsort_order, lengths

def flatten_label_batch(batch):
    batch_size = batch.shape[0]
    label_lengths = torch.zeros(batch_size)
    for idx in range(batch_size):
        label_lengths[idx] = batch[idx].shape[0]
    all_labels = torch.cat([torch.from_numpy(label) for label in batch], dim=0) + 1

    return all_labels, label_lengths


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            downsample = None
        self.downsample = downsample

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class PhonemeModel(nn.Module):
    def __init__(self):
        super(PhonemeModel, self).__init__()
        self.cnn = ResidualBlock(1, 4, stride=1, kernel_size=5, padding=2)
        self.rnn = nn.LSTM(input_size=160, hidden_size=256, num_layers=3, bidirectional=True, dropout=0.5)
        # self.wdrnn = WeightDrop(self.rnn, ['weight_hh_l0', 'weight_hh_l1', 'weight_hh_l2', 'weight_hh_l0_reverse', 'weight_hh_l1_reverse', 'weight_hh_l2_reverse'], dropout=0.5)
        self.out_linear = nn.Sequential(nn.Linear(512, 47))

        self.init_hidden = nn.Parameter(torch.zeros(6, 1, 256), requires_grad=True)
        self.init_cell = nn.Parameter(torch.zeros(6, 1, 256), requires_grad=True)
        self.init_weights()
        # self.rnn.flatten_parameters()

        self.ctc = CTCLoss()
        self.label_map = [' '] + PHONEME_MAP
        self.decoder = CTCBeamDecoder(labels=self.label_map)

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            print('param:', name)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for layer in self.out_linear:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, inputs):
        # packed, sort_order, unsort_order, lengths = sort_pad_pack_batch(inputs)
        padded, sort_order, unsort_order, lengths = sort_pad_batch(inputs)
        outputs = self.cnn(torch.unsqueeze(padded, dim=1))
        # outputs = torch.squeeze(outputs, dim=1)
        outputs = torch.transpose(outputs, 1, 2)
        outputs = outputs.contiguous().view(outputs.shape[0], outputs.shape[1], -1)
        packed = pack_padded_sequence(outputs, lengths[sort_order], batch_first=True)
        h_0, c_0 = self.init_hidden.repeat(1, lengths.shape[0], 1), self.init_cell.repeat(1, lengths.shape[0], 1)
        outputs, _ = self.rnn(packed, (h_0, c_0))
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.out_linear(outputs)
        outputs = outputs[unsort_order]

        return outputs, lengths

    def loss(self, inputs, targets):
        outputs, act_lengths = self.forward(inputs)
        labels, label_lengths = flatten_label_batch(targets)
        a, b, c, d = torch.transpose(outputs, 0, 1), labels.int(), torch.from_numpy(act_lengths).int(), label_lengths.int()
        loss = self.ctc(a, b, c, d)

        return loss

    def decode(self, inputs, targets=None):
        outputs, act_lengths = self.forward(inputs)
        probs = F.softmax(outputs, dim=2)

        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=torch.from_numpy(act_lengths).int())
        preds = []
        for i in range(output.shape[0]):
            pred = ''.join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
            preds.append(pred)

        if targets is None:
            print(len(preds))
            return preds

        loss = 0
        print(preds[0], ''.join(self.label_map[l] for l in targets[0]+1))
        for pred, target in zip(preds, targets):
            label = ''.join(self.label_map[l] for l in target+1)
            dist = L.distance(pred, label)
            loss += dist

        return preds, loss