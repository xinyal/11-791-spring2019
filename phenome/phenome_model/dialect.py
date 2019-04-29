import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import librosa
import os
import random

from torch.optim import Adam
from model import PhonemeModel, sort_pad_pack_batch, sort_pad_batch, ResidualBlock
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader


def predict(mfcc_path, model):
    mfcc = np.load(mfcc_path)
    mfcc = mfcc.T.reshape(1, mfcc.shape[1], mfcc.shape[0])
    pred = torch.argmax(model(mfcc)[0]).item()
    return pred

class DialectModel(nn.Module):
    def __init__(self, max_frame_size=1000, use_cuda=True):
        super(DialectModel, self).__init__()
        
        # self.phenome_model = torch.load(pt)
        # self.max_frame_size = max_frame_size
        # self.cnn = nn.Conv1d(512, 512, kernel_size=5, stride=2)
        # self.bn = nn.BatchNorm1d(512)
        # self.pool = nn.MaxPool1d(self.max_frame_size // 2, padding=0)
        # self.linear = nn.Linear(512, 6)

        self.res1 = ResidualBlock(1, 32, stride=2, kernel_size=5, padding=2)
        # self.res2 = ResidualBlock(32, 64, stride=2, kernel_size=5, padding=2)
        # self.res3 = ResidualBlock(64, 128, stride=2, kernel_size=5, padding=2)
        self.pool = nn.AvgPool2d((500, 1), padding=0)
        self.linear1 = nn.Linear(640, 256)
        self.linear2 = nn.Linear(256, 6)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        
    def phenome_feature(self, inputs):
        # packed, sort_order, unsort_order, lengths = sort_pad_pack_batch(inputs)
        padded, sort_order, unsort_order, lengths = sort_pad_batch(inputs)
        outputs = self.phenome_model.cnn(torch.unsqueeze(padded, dim=1))
        # outputs = torch.squeeze(outputs, dim=1)
        outputs = torch.transpose(outputs, 1, 2)
        outputs = outputs.contiguous().view(outputs.shape[0], outputs.shape[1], -1)
        packed = pack_padded_sequence(outputs, lengths[sort_order], batch_first=True)
        h_0, c_0 = self.phenome_model.init_hidden.repeat(1, lengths.shape[0], 1), self.phenome_model.init_cell.repeat(1, lengths.shape[0], 1)
        outputs, _ = self.phenome_model.rnn(packed, (h_0, c_0))
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # outputs = self.phenome_model.out_linear(outputs)
        outputs = outputs[unsort_order]

        return outputs, lengths

    def forward(self, inputs):
        # outputs, lengths = self.phenome_feature(inputs)
        # # print(outputs.shape)
        # outputs = self.cnn(outputs.transpose(1, 2))
        # outputs = F.relu(self.bn(outputs))
        # # print(outputs.shape)
        # outputs = self.pool(outputs).squeeze()
        # # print(outputs.shape)
        # pred = self.linear(outputs)

        out = torch.unsqueeze(torch.from_numpy(inputs).float(), dim=1)
        if self.use_cuda:
            out = out.cuda()
        out = self.res1(out)
        # out = self.res2(out)
        # out = self.res3(out)
        # print(out.shape)
        out = self.pool(out)
        # print(out.shape)
        out = out.view(inputs.shape[0], -1)
        out = self.linear1(out)
        pred = self.linear2(F.relu(out))

        return pred

    def loss(self, inputs, labels):
        pred = self.forward(inputs)
        loss = F.cross_entropy(pred, labels)
        acc = torch.mean(torch.eq(torch.argmax(pred, dim=1), labels).float())

        return loss, acc


class DialectDataloader(DataLoader):
    def __init__(self, data_path, partition_path, batch_size, frame_size, shuffle=True):
        print('loading data from {}'.format(partition_path))
        self.dataset = []
        self.labels = []
        self.label_dict = {'at': 0, 'mi': 1, 'ne': 2, 'no': 3, 'so': 4, 'we': 5}
        with open(partition_path, 'r') as f:
            lines = f.readlines()
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                fname = line[line.rfind('/')+1 : line.find('.')]
                if fname[3] == 'E':
                    continue
                mfcc_path = os.path.join(data_path, fname+'.npy')
                mfcc = np.load(mfcc_path)
                if mfcc.shape[1] < frame_size:
                    mfcc = np.repeat(mfcc, repeats=frame_size // mfcc.shape[1] + 1, axis=1)
                    mfcc = mfcc[:, :frame_size]
                elif mfcc.shape[1] > frame_size:
                    mfcc = mfcc[:, :frame_size]
                self.dataset.append(mfcc.T)
                self.labels.append(self.label_dict[fname[:2]])

        # self.dataset = np.asarray(self.dataset)
        self.labels = torch.from_numpy(np.asarray(self.labels)).cuda()

        self.batch_size = batch_size
        self.shuffle = shuffle
        num_batches = len(self.dataset) // batch_size
        self.batches = [(i * batch_size, (i+1) * batch_size) for i in range(num_batches)]
        if len(self.dataset) > num_batches * batch_size:
            self.batches.append((num_batches * batch_size, len(self.dataset)))

        print('{} snippets loaded.'.format(len(self.dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            data = self.dataset[batch[0]: batch[1]]
            label = self.labels[batch[0]: batch[1]]
            yield np.asarray(data), label

    def __len__(self):
        return len(self.batches)


if __name__ == '__main__':

    model = DialectModel()

    train_loader = DialectDataloader('../../../MFCC/', '../../en_data/filepaths.train', 32, 1000)
    dev_loader = DialectDataloader('../../../MFCC/', '../../en_data/filepaths.dev', 32, 1000, shuffle=False)

    optimizer = Adam(model.parameters(), lr=1e-3)

    NUM_EPOCH = 10

    for epoch in range(NUM_EPOCH):
        model.train()
        train_loss = 0
        train_acc = 0

        for data, label in train_loader:
            optimizer.zero_grad()
            loss, acc = model.loss(data, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        print('epoch {} train loss: {} acc: {}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))

        with torch.no_grad():
            model.eval()
            dev_acc = 0
            fout = open('dev_{}_result.txt'.format(epoch), 'w')
            for data, label in dev_loader:
                pred = model.forward(data)
                pred = torch.argmax(pred, dim=1)
                acc = torch.mean(torch.eq(pred, label).float())
                dev_acc += acc
                for p, l in zip(pred, label):
                    fout.write('{} {}\n'.format(p, l))
            fout.close()
            print('epoch {} dev acc: {}'.format(epoch, dev_acc / len(dev_loader)))