import torch
from PIL import Image
import numpy as np
import random

def corr_compute(original, np_frame):
	 return np.corrcoef(original, np_frame)[0,1]

def weight_decay(model, lamb):
	for n, p in model.named_parameters():
		p.grad.data.add_(lamb*p.data)

def watermark_mask(model, P):
	mask = {}
	for n, p in model.named_parameters():
		mask[n] = torch.rand(p.data.size()) < P
	return mask

def freeze_watermark(model, mask):
	for n, p in model.named_parameters():
		p.grad.data.mul_(mask[n].type(torch.float))

def bury_watermark(model, raw_watermark, P, SF):
	arch_boundaries = {}
	names = []
	tot_params = []
	for n, p in model.named_parameters():
		arch_boundaries[n] = p.data.numel()-1
		names.append(n)
		tot_params.append(p.data.numel())
	N = 0
	p = np.zeros(len(names))
	j = 0
	for i in tot_params:
		N += i
		p[j] = i
		j += 1
	watermark = watermark_mask(model, P)
	model_dict = model.state_dict()
	###now generate position and mask
	index = np.zeros([np.size(raw_watermark, 0), np.size(raw_watermark, 1), np.size(raw_watermark, 2), 2], int)
	for i in range(raw_watermark.shape[0]):
		for j in range(raw_watermark.shape[1]):
			for k in range(raw_watermark.shape[2]):
				index[i, j, k, 0] = np.random.choice(len(names), p=p/N)
				N -= 1
				p[index[i, j, k, 0]] -= 1
				index[i, j, k, 1] = random.randrange(arch_boundaries[names[index[i, j, k, 0]]])##parameter index extracted
				while watermark[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]] == 0:##already used, to be resampled
					index[i, j, k, 1] = random.randrange(arch_boundaries[names[index[i, j, k, 0]]])##parameter index extracted
				model_dict[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]]*= 0
				model_dict[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]]+= (raw_watermark[i,j,k])
				watermark[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]] = 0##mask this value forever
	return index, names, watermark

def get_watermark(path, SF):
	im_frame = Image.open(path).convert("RGB")
	width, height = im_frame.size
	np_frame = np.array(im_frame.getdata()).reshape(height,width,3)
	np_frame = (np_frame/255.0 *2 - 1)/SF
	return np_frame

def exhume_watermark(model, index, names, shap):
	model_dict = model.state_dict()
	retrieved_watermark = np.ones([np.size(index, 0), np.size(index, 1), np.size(index, 2)], float)*255
	for i in range(shap[0]):
		for j in range(shap[1]):
			for k in range(shap[2]):
				retrieved_watermark[i,j,k] = model_dict[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]]
	return retrieved_watermark

def generate_mask(model, index, names, shap):
	model_dict = model.state_dict()
	retrieved_watermark = np.ones([np.size(index, 0), np.size(index, 1), np.size(index, 2)], float)*255
	mask = {}
	for n, p in model.named_parameters():
		mask[n] = torch.ones(p.data.size())
	for i in range(shap[0]):
		for j in range(shap[1]):
			for k in range(shap[2]):
				mask[names[index[i, j, k, 0]]].view(-1)[index[i, j, k, 1]] = 0
	return mask

def save_watermark(path, np_frame, SF):
	np_frame = (np_frame * SF + 1)/2.0  * 255
	np_frame = np_frame.astype('uint8')
	im_extracted = Image.fromarray(np_frame).convert('RGB')
	im_extracted.save(path, "PNG")
