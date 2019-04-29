from baseline_nsp import *

def pred(filename):
	INV_LABEL_DICT = {0:'Mid-Atlantic', 1:'Midland', 2:'New England', 3:'North', 4:'South', 5:'West'}

	# loading model
	if torch.cuda.is_available():
		model = encoder().cuda()
	else:
		model = encoder()

	model.load_state_dict(torch.load("./src/model_checkpoint.pt", map_location='cpu'))
	model.eval()

	with torch.no_grad():
		feat_input = np.load(filename)
		feat_input = feat_input.astype(np.float32).transpose() # 196

		feat_input = torch.LongTensor(feat_input)
		l = torch.LongTensor([int(feat_input.size(0))]) # 196

		if torch.cuda.is_available():
			feat_input = feat_input.cuda()
			l = l.cuda()
		feat_input = feat_input.view((feat_input.size(0), 1, feat_input.size(1)))

		output = model(feat_input, l)
		prediction = output.data.cpu().detach().numpy().argmax(1).item()
		prediction = INV_LABEL_DICT[prediction]

	return prediction
