import os
from os.path import dirname, abspath
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence
import numpy
import pickle
import pdb
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
import math

REPO_PATH = dirname(dirname(abspath(__file__))) # /home/kristina/desire-directory #REPO_PATH = os.path.abspath(__file__ + "/..")
DATA_PATH = os.path.abspath(REPO_PATH + "/../nsp_wav")
ID_PATH = os.path.abspath(REPO_PATH + "/en_data")
FEATURE_PATH = os.path.join(DATA_PATH + "/MFCC")
### Help:  Piazza posts: https://piazza.com/class/j9xdaodf6p1443?cid=2774
# hidden = 256
batch_size = 21

def get_label(fname):
    regions = ['at', 'mi', 'ne', 'no', 'so', 'we']
    if any(fname.startswith(region) for region in regions):
        return fname[:2]
    return None

LABEL_DICT = {'at':0, 'mi':1, 'ne':2, 'no':3, 'so':4, 'we':5}

#######################
# Function Definitions
#######################
def process_input(input_type, input_filepaths=ID_PATH, feat_dir=FEATURE_PATH):
    # Process input # input_type = {'dev', 'train'}
    # save input and output .npy files in processing dir
    print("Processing {}".format(input_type))
    with open(os.path.join(input_filepaths, 'filepaths.{}.short'.format(input_type))) as f:
        audio_array = []
        label_array = []
        for line in f:
            if line[3] == 'E': continue
            line = line.strip()
            input_file = os.path.join(feat_dir, line + '.npy')
            A = np.load(input_file)
            audio_array.append(A.astype(np.float32))
            label_array.append(get_label(line))  # assume line is filename
    #DATA_PATH+'{}_audio.pkl'.format(input_type)
    pickle.dump(audio_array, open('{}_audio.pkl'.format(input_type), "wb"), protocol = 2)
    pickle.dump(label_array, open('{}_label.pkl'.format(input_type), "wb"), protocol = 2)


class NSP(Dataset):
    def __init__(self, audio, label):
        feat = [a.transpose() for a in audio]
        lab = [LABEL_DICT[l] for l in label]
        d = {"audio":feat, "label":lab}
        self.data = pd.DataFrame(data=d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f = self.data.iloc[idx,0]
        l = self.data.iloc[idx,1]
        sample = {'audio': torch.LongTensor(f), 'label': torch.LongTensor([int(l)])}
        return (sample)


def collate(batch): # batch = [trainset[5274],trainset[3274],trainset[1]]
    sorted_batch = sorted(batch, key=lambda x: x['audio'].size(0), reverse=True)
    sorted_audio = [sb['audio'] for sb in sorted_batch]
    sorted_audio_l = [sa.size(0) for sa in sorted_audio]
    sorted_tag = [sb['label'] for sb in sorted_batch]

    padded_audio = pad_sequence(sequences=sorted_audio)#, batch_first = True) # try padding_value = 1
    return (padded_audio, torch.LongTensor(sorted_tag), torch.LongTensor(sorted_audio_l))


class encoder(nn.Module): # https://github.com/srvk/Yunitator/blob/master/Yunitator/Net.py
    def __init__(self, input_dim=40, hidden_dim=10, output_size=6):
        super(encoder, self).__init__()
        self.gru = nn.GRU(input_size = input_dim, hidden_size=hidden_dim,
            num_layers=2, bidirectional = True)#, batch_first = True)
        #self.gru = nn.GRU(input_size=input_dim, )
        #self.fc2 = nn.Linear(hidden_dim, 3)
        self.fc = nn.Linear(hidden_dim * 2, output_size) # Bidirectional, so the size of the output is 2*nHidden

    def forward(self, seq, lens):
        #seq_p = seq.view(seq.size(2), seq.size(0), seq.size(1)) # [3, 494, 40] to [40, 3, 494]
        #seq_p = seq
        #self.gru.input_size = seq_p.size(-1)
        # https://discuss.pytorch.org/t/expected-object-of-scalar-type-long-but-got-scalar-type-float-for-argument-2-mat2/34063/2
        # https://discuss.pytorch.org/t/how-to-cast-a-tensor-to-another-type/2713/8
        #self.gru.input_size = seq_p.size(-1) # expect input_size = input.size(-1)
        seq = seq.type(torch.FloatTensor) # torch.cuda.FloatTensor
        if torch.cuda.is_available():
            seq = seq.cuda()
        #pdb.set_trace()
        x = self.gru(seq)[0] # a tuple of 2 tensor. [40,3,1000] and [4,40,500]. [fs, bs, 2*hs] [2*nl, fs, hs]
        #x = pack_padded_sequence(seq, lens, batch_first=True)
        res = PackedSequence(F.softmax(self.fc(x[0]), dim=-1), x[1])
        return res

##############
# Begin main()
##############

def main():
    if not os.path.exists("train_audio.pkl"):
        print ("dumping train data to a cucumber...")
        process_input("train")
    if not os.path.exists("dev_audio.pkl"):
        print ("dumping dev data to a cucumber...")
        process_input("dev")

    with open("train_audio.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("train_label.pkl", "rb") as f:
        train_labels = pickle.load(f)
    with open("dev_audio.pkl", "rb") as f:
        dev_data = pickle.load(f)
    with open("dev_label.pkl", "rb") as f:
        dev_labels = pickle.load(f)

    test_data = dev_data[len(dev_data)//2:]
    test_labels = dev_labels[len(dev_labels)//2:]
    dev_data = dev_data[:len(dev_data)//2]
    dev_labels = dev_labels[:len(dev_labels)//2]

    #print (len(train_data), len(train_labels), len(dev_data), len(dev_labels), len(test_data), len(test_labels))
    #max_lens = [max([td.shape[1] for td in train_data]), max([td.shape[1] for td in test_data]),
    #            max([td.shape[1] for td in dev_data])]
    #max_l = max(max_lens)
    #print (sorted(list(set([td.shape[1] for td in train_data])), reverse=True))

    trainset = NSP(train_data, train_labels)
    devset = NSP(dev_data, dev_labels)
    testset = NSP(test_data, test_labels)

    train_loader = data_utils.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn = collate)
    dev_loader = data_utils.DataLoader(devset, batch_size=1, shuffle=False, collate_fn = collate)
    test_loader = data_utils.DataLoader(testset, batch_size=1, shuffle=False, collate_fn = collate)

    print("The target size is ", 6)

    num_epochs = 10
    best_acc = -1

    if torch.cuda.is_available():
        print ("cuda!")
        #model = encoder()
        model = encoder().cuda()
    else:
        model = encoder()

    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    lr_scheduler = StepLR(optimizer, step_size = 1, gamma = 0.1)
    #optimizer = torch.optim.SGD(list(model.parameters()), lr=0.01)
    num_batch = math.ceil(len(train_data) / batch_size)

    _patience = patience = 5
    num_trials = 5
    cache_path_model = "./model_checkpoint.pt"
    #cache_path_model_batch = "./batch_checkpoint.pt"
    cache_path_optim = "./optim_checkpoint.pt"


    for epoch in range(num_epochs):
        t = tqdm(train_loader, total=num_batch)
        for batch in t:
            model.zero_grad()
            x,y,l = batch
            x = x.cuda()
            y = y.cuda()
            l = l.cuda()
            output = model(x,l)
            #print (output.data.size(), y.size())
            loss = loss_func(output.data, y)
            loss.backward()
            optimizer.step()
            print("Current loss: {}".format(loss.item()))
            t.set_description("Current loss: {}".format(loss.item()))

        with torch.no_grad():
            pred = []
            truth = []
            for batch in tqdm(dev_loader):
                model.zero_grad()
                x_v, y_v, l = batch
                x_v = x_v.cuda()
                y_v = y_v.cuda()
                l = l.cuda()
                output = model(x_v, l)
                truth.append(y_v.cpu().numpy())
                pred.append(output.data.cpu().detach().numpy().argmax(1))
                #pred.append(output.cpu().numpy().argmax(1))
            truth = np.concatenate(truth, axis=0)
            pred = np.concatenate(pred, axis=0)
            acc = accuracy_score(truth, pred)
            print ("Dev set accuracy is: {}".format(acc)) #print (f"Dev set accuracy is: {acc}")

        if acc > best_acc:
            best_acc = acc
            patience = _patience
            print ("Found a new best model! Saving model state and optimizer state to disk")
            torch.save(model.state_dict(), cache_path_model)
            torch.save(optimizer.state_dict(), cache_path_optim)
        else:
            patience -= 1
        print ("Current patience: {}".format(patience)) #print (f"Current patience: {patience}")

        if patience <= 0:
            num_trials -= 1
            patience = _patience
            print ("Starting a new trial with the next best model and optimizer")
            model.load_state_dict(torch.load(cache_path_model))
            optimizer.load_state_dict(torch.load(cache_path_optim))
            lr_scheduler.step()


        if num_trials <= 0:
            print ("Early stopping, running out of trials")
            break

if __name__ == '__main__':
    main()
