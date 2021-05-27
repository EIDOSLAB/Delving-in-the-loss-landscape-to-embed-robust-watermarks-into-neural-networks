import torch

def reinitialize_neighbor(model, reference, mask, std, device):
	N2pams = reference.state_dict()
	for n, p in model.named_parameters():
		p2 = N2pams[n]
		p.data.copy_(p2.data)
		rev_mask = (-mask[n].type(torch.float)+1) * torch.randn(mask[n].size(), device = device)*std
		p.data.add_(rev_mask)

def update_center(model, neighbor_models, mask, lamb, device):
	for n, p in model.named_parameters():
		update = torch.zeros(p.data.size(), device = device)
		for rep in range(len(neighbor_models)):
			update += neighbor_models[rep].state_dict()[n].data
		update /= (len(neighbor_models))
		update = update*(mask[n].type(torch.float))
		p.data.add_(-lamb*update)
