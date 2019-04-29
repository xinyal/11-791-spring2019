from baseline_nsp import *

def pred(filename):
	INV_LABEL_DICT = {0:'at', 1:'mi', 2:'ne', 3:'no', 4:'so', 5:'we'}

	# loading model
	if torch.cuda.is_available():
        model = encoder().cuda()
    else:
        model = encoder()

    model.load_state_dict(torch.load("./model_checkpoint.pt"))
    model.eval()

    with torch.no_grad():
		feat_input = np.load(filename)
		feat_input = feat_input.astype(np.float32).transpose() # 196 

		feat_input = torch.LongTensor(feat_input) 
		l = torch.LongTensor([int(feat_input.size(0))]) # 196

		feat_input = feat_input.cuda()
		l = l.cuda()
		feat_input = feat_input.view((feat_input.size(0), 1, feat_input.size(1)))

		output = model(feat_input, l)
		prediction = output.data.cpu().detach().numpy().argmax(1).item()
		prediction = INV_LABEL_DICT[prediction]

	return prediction
