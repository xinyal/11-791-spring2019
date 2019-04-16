from __future__ import print_function

import torch
from torch.optim import Adam

from model import PhonemeModel
from data import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import datetime


def run():
    train_loader = load_train()
    dev_loader = load_dev()
    test_loader = load_test()

    model = PhonemeModel()
    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    NUM_EPOCH = 30
    best_dev = 1e10
    runid = str(datetime.datetime.now())

    for epoch in range(NUM_EPOCH):
        model.train()

        epoch_loss = 0

        for b_id, batch in enumerate(train_loader):
            data, label = batch
            optimizer.zero_grad()
            loss = model.loss(data, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print('epoch {} train loss: {}'.format(epoch, epoch_loss / 24724.0))

        if ((epoch + 1) % 2 == 0 and epoch > 9) or ((epoch + 1) % 5 == 0 and epoch <= 9):
            model.eval()
            with torch.no_grad():
                dev_loss = 0
                for batch in dev_loader:
                    data, label = batch
                    preds, loss = model.decode(data, label)
                    dev_loss += loss
                print('epoch {} dev loss: {}'.format(epoch, dev_loss / 1106.0))

                # if dev_loss < best_dev:
                #     best_dev = dev_loss
                #     print('run test')
                #     all_preds = []
                #     for batch in test_loader:
                #         preds = model.decode(batch)
                #         all_preds += preds
                #     with open('{}-{}.csv'.format(runid, epoch), 'w') as f:
                #         f.write('Id,Predicted\n')
                #         for i, pred in enumerate(all_preds):
                #             f.write('{},{}\n'.format(i, pred))

            torch.save(model, 'epoch-{}.pt'.format(epoch))


if __name__ == '__main__':
    run()