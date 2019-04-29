import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


# from dev output file, calculate accuracy, precision, recall, f1
region_to_num = {'at': 0, 'mi': 1, 'ne': 2, 'no': 3, 'so': 4, 'we': 5}
num_to_region = {val: key for key, val in region_to_num.iteritems()}


def show_confusion(truths, preds):
	classes = region_to_num.keys()
	word_truths = [num_to_region[item] for item in truths]
	word_preds = [num_to_region[item] for item in preds]
	cm = confusion_matrix(word_truths, word_preds, labels=classes)

	# normalize
	# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	np.set_printoptions(precision=2)
	plt.figure()
	cmap = plt.cm.Blues
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title('Confusion Matrix of Dialects')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	# fmt = '.2f'
	fmt = 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True Regions')
	plt.xlabel('Predicted Regions')
	plt.show()


def calc_fpr(truths, preds):
	accuracy = accuracy_score(truths, preds)

	prf = precision_recall_fscore_support(truths, preds, average='weighted')
	precision = prf[0]
	recall = prf[1]
	f_measure = prf[2]

	print 'accuracy', accuracy
	print 'precision', precision
	print 'recall', recall
	print 'f_measure', f_measure


def get_labels(output_file):
	with open(output_file, 'r') as f:
		lines = [line.strip() for line in f.readlines()]

	preds = [int(line.split()[0]) for line in lines]
	truths = [int(line.split()[1]) for line in lines]
	return truths, preds


if __name__ == '__main__':
	# truths, preds = get_labels('en_data/dev_0_result_0422.txt')
	truths, preds = get_labels('en_data/dev_1_result_0422.txt')
	calc_fpr(truths, preds)
	# accuracy 0.233
	# precision 0.197
	# recall 0.233
	# f_measure 0.197

	show_confusion(truths, preds)
