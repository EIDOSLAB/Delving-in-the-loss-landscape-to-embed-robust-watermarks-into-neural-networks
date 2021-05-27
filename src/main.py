import torch
import argparse
import os
from torch import autograd, nn, optim
import torchvision
from torchvision import transforms, datasets
from watermarklib import *
from replicalib import *
from models import *


def train(model, neighbor_models, optimizer, neighbor_optimizer, criterion, train_loader, mask, lamb, std, device):
	model.train()
	for data, target in train_loader:
		data = data.to(device)
		target = target.to(device)

		if lamb != 0:
			for net in neighbor_models:
				reinitialize_neighbor(net, model, mask, std, device)

		output= model(data)
		optimizer.zero_grad()
		loss = criterion(output, target)
		loss.backward()
		weight_decay(model, 0.0005)

		freeze_watermark(model, mask)
		optimizer.step()
		if lamb != 0:
			for net_idx in range(len(neighbor_models)):
				output= neighbor_models[net_idx](data)
				neighbor_optimizer[net_idx].zero_grad()
				loss = -criterion(output, target)###because we MAXIMIZE it!
				loss.backward()
				#freeze_watermark(net[net_idx], mask)####not necessary for 1 iteration computation!
				neighbor_optimizer[net_idx].step()
			update_center(model, neighbor_models, mask, lamb, device)


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PSP regu')
	parser.add_argument('--batch_size', type=int, default=100, metavar='N',
	                    help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=300, metavar='N',
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
	                    help='learning rate (default: 0.01)')
	parser.add_argument('--lamb', type=float, default=0.000001)
	parser.add_argument('--R', type=int, default=4, help='number of replicas')
	parser.add_argument('--R_stdev', type=float, default=0.01, help='STD of the noise to apply to replicas')
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--datapath', default=f'{os.path.expanduser("~")}/data/', help='path where dataset is stored')
	parser.add_argument('--arch', default='resnet32')
	parser.add_argument('--dataset', default='cifar10')
	parser.add_argument('--path_watermark', default='watermark_example.png')
	parser.add_argument('--SF', default=0.1, help='scale factor to apply to the watermark to be embed in the model')
	parser.add_argument('--save_private', default='private.pt')
	parser.add_argument('--save_model', default='model.pt')
	parser.add_argument('--seed', default=1)
	parser.add_argument('--device', default='cpu')

	args = parser.parse_args()
	device = torch.device(args.device)
	np.random.seed(args.seed)

	##extract image to use as watermark
	loss_function = nn.CrossEntropyLoss()
	
	print('==> Preparing data..')
	if args.dataset == 'cifar10':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		trainset = torchvision.datasets.CIFAR10(root=args.datapath, train=True, download=True, transform=transform_train)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

		testset = torchvision.datasets.CIFAR10(root=args.datapath, train=False, download=True, transform=transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

		if args.arch == 'resnet32':
			model = resnet32('A').to(device)
			neighbor_models = [resnet32('A').to(device) for i in range(args.R)]
		elif args.arch == 'ALL-CNN-C':
			model = ALL_CNN_C().to(device)
			neighbor_models = [ALL_CNN_C().to(device) for i in range(args.R)]
		else:
			print('ERROR: Not implemented')
			return
		print(len(neighbor_models))
		optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250,350], gamma=0.1)

	elif args.dataset == 'mnist':
		train_loader = torch.utils.data.DataLoader(
	    	datasets.MNIST(data_folder, train=True, download=True,
	                   transform=transforms.Compose([
	                       transforms.ToTensor(),
	                       transforms.Normalize((0.1307,), (0.3081,))
	                   ])),
	    	batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory=True)
		test_loader = torch.utils.data.DataLoader(
			datasets.MNIST(data_folder, train=False, transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.1307,), (0.3081,))
						])),
			batch_size=args.batch_size, shuffle=False, num_workers = 4, pin_memory=True)
		if args.arch == 'lenet5':
			model = LeNet5().to(device)
			optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
			neighbor_models = [LeNet5().to(device) for i in range(args.R)]
			scheduler = None
		else:
			print('ERROR: Not implemented')
			return
	else:
		print('ERROR: Not implemented')
		return

	##make indices, allocate mask and initialized buried parameters
	raw_watermark = get_watermark(args.path_watermark, args.SF)
	use_watermark = {}

	index, names, watermark = bury_watermark(model, raw_watermark, 1, args.SF)

	for n, p in model.named_parameters():
	    use_watermark[n] = watermark[n].to(device)
	print(len(neighbor_models))
	neighbor_optimizer = [optim.SGD(neighbor_models[i].parameters(), lr=1, weight_decay=0.0) for i in range(args.R)]


	for epoch in range(args.epochs):
		print("Train epoch: {}".format(epoch))
		train(model, neighbor_models, optimizer, neighbor_optimizer, loss_function, train_loader, use_watermark, args.lamb, args.R_stdev, device)
		test(model, device, test_loader, loss_function)
		if scheduler is not None:
			scheduler.step()

	#retrieved_watermark = exhume_watermark(model, raw_watermark, index, names)
	#save_watermark('exhumed_watermark.png', retrieved_watermark, SF)
	torch.save([index, names, raw_watermark.shape], args.name_private)
	torch.save(model, args.save_model)

if __name__ == '__main__':
    main()