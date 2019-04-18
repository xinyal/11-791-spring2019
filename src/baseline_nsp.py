import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.metrics import classification_report
from torch.utils.data import Dataset


### Help:  Piazza posts: https://piazza.com/class/j9xdaodf6p1443?cid=2774
# num_classes = 6
# input_dim = 50
# hidden = 256
batch_size = 4
# print_flag = 1


def get_label(fname):
	regions = ['at', 'mi', 'ne', 'no', 'so', 'we']
	if any(fname.startswith(region) for region in regions):
		return fname[:2]

	return None


# Process input
# input_type = {'dev', 'train'}
# save input and output .npy files in processing dir
def process_input(input_type, input_filepaths, feat_dir):
	# input_type = 'dev'
	# input_filepaths = 'filepaths.dev.short'
	# feat_dir = './MFCC/'
	print("Processing {}".format(input_type))
	f = open(input_filepaths)
	input_array = []
	output_array = []
	for line in f:
		line = line.strip()
		input_file = os.path.join(feat_dir, line + '.npy')
		A = np.load(input_file)
		a = A['arr_0']
		inp = np.mean(a[4], axis=0)
		input_array.append(inp.astype(np.float32))
		output_array.append(get_label(line))  # assume line is filename

	np.save('{}_input.npy'.format(input_type), input_array)
	np.save('{}_output.npy'.format(input_type), output_array)


#TODO: check relative paths
process_input('dev', 'filepaths.dev.short', 'MFCC/')
process_input('train', 'filepaths.train.short', 'MFCC/')

train_input_array = np.load('train_input.npy')
train_output_array = np.load('train_output.npy')
devel_input_array = np.load('dev_input.npy')
devel_output_array = np.load('dev_output.npy')


class COMPARE(Dataset):

	def __init__(self, A, B):

		self.input = A
		self.output = B
		#print(B)

	def __len__(self):
		return len(self.input)

	def __getitem__(self, idx):
		return self.input[idx], self.output[idx]


trainset = COMPARE(train_input_array, train_output_array)
devset = COMPARE(devel_input_array, devel_output_array)
train_loader = data_utils.DataLoader(trainset, batch_size=batch_size, shuffle=True)
dev_loader = data_utils.DataLoader(devset, batch_size=1, shuffle=False)


class encoder(nn.Module):

	def __init__(self, input_dim, hidden_dim):
		super(encoder, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, 3)

	def forward(self, x):

		x = torch.tanh(self.fc1(x))
		return self.fc2(x)

#TODO: check defaults
embedding_dim = 6  # used to be 3
hidden_dim = 128
vocab_size = 6  # used to be 3
target_size = vocab_size
input_dim = 512
print("The target size is ", target_size)
baseline_encoder = encoder(input_dim, hidden_dim)
if torch.cuda.is_available():
	baseline_encoder + baseline_encoder.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(baseline_encoder.parameters()), lr=0.01)
objective = nn.CrossEntropyLoss()


def train():
	total_loss = 0
	for ctr, t in enumerate(train_loader):
		a, b = t
		#print("Shape of input, output: ", a.shape, b.shape)
		#print("Type of input, output is ", a.data.type(), b.data.type())
		if torch.cuda.is_available():
			a, b = a.cuda(), b.cuda()
		pred = baseline_encoder(a.float())
		#print("Shape of encoder output:", pred.shape)
		#print("Batch Done")
		loss = criterion(pred, b)
		total_loss += loss.cpu().data.numpy()
		#if ctr % 100 == 1:
		#	 print("Loss after ", ctr, "batches: ", total_loss/(ctr+1))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print("Train Loss is: ", total_loss/ctr)
	test()
	print('\n')


def test():
	baseline_encoder.eval()
	total_loss = 0
	ytrue = []
	ypred = []
	for ctr, t in enumerate(dev_loader):
		a, b = t
		ytrue.append(b.data.numpy()[0])
		if torch.cuda.is_available():
			a, b = a.cuda(), b.cuda()
		pred = baseline_encoder(a.float())
		loss = criterion(pred, b)
		total_loss += loss.cpu().data.numpy()
		prediction = np.argmax(pred.cpu().data.numpy())
		ypred.append(prediction)
		#if ctr % 200 == 1:
		#	print ("Prediction, Original: ", prediction, b.cpu().data.numpy())

	#print(ytrue[0:10], ypred[0:10])
	print(classification_report(ytrue, ypred))
	print("Test Loss is: ", total_loss/ctr)
	baseline_encoder.train()

for epoch in range(10):
	print("Running epoch ", epoch)
	train()
