import math
import torch

import torch.nn.functional as F
from torch import nn


class LeNet5(nn.Module):
	def __init__(self):
		super(LeNet5, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, bias=True)
		self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, bias=True)
		self.fc1 = nn.Linear(4 * 4 * 50, 500, bias=True)
		self.fc2 = nn.Linear(500, 10, bias=True)
		self.conv1.weight.data.normal_(0.0,0.01)
		self.conv2.weight.data.normal_(0.0,0.01)
		self.fc1.weight.data.normal_(0.0,0.01)
		self.fc2.weight.data.normal_(0.0,0.01)

	def forward(self, img):
		# no watermark on the first layer
		x = img
		z = self.conv1(x)
		x = F.max_pool2d(F.relu(z), 2)

		z = self.conv2(x)
		x = F.max_pool2d(F.relu(z), 2)
		x = x.view(img.size(0), -1)

		z = self.fc1(x)
		x = F.relu(z)
		output = self.fc2(x)
		return output

def test(model, device, test_loader, criterion):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += criterion(output, target)
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	    test_loss, correct, len(test_loader.dataset),
	    100. * correct / len(test_loader.dataset)))
	
	return test_loss.item(), 100.0 - (100. * correct / len(test_loader.dataset))

def get_loss(model, data_loader, loss):
	model.eval()
	this_loss = 0
	for data, target in data_loader:
		output, x, z = model(data)
		this_loss = this_loss + loss(output, target).item()
	return this_loss


class ALL_CNN_C(nn.Module):

	def __init__(self, num_classes = 10):
		super(ALL_CNN_C, self).__init__()
		self.model_name = 'ALL_CNN_C'

		self.dp0 = nn.Dropout2d(p = 0.2)

		self.conv1 = nn.Conv2d(3, 96, 3, padding = 1)

		self.conv2 = nn.Conv2d(96, 96, 3, padding = 1)

		self.conv3 = nn.Conv2d(96, 96, 3, stride = 2, padding = 1)
		self.dp1 = nn.Dropout2d(p = 0.5)

		self.conv4 = nn.Conv2d(96, 192, 3, padding = 1)

		self.conv5 = nn.Conv2d(192, 192, 3, padding = 1)

		self.conv6 = nn.Conv2d(192, 192, 3, stride = 2, padding = 1)
		self.dp2 = nn.Dropout2d(p = 0.5)

		self.conv7 = nn.Conv2d(192, 192, 3, padding = 0)

		self.conv8 = nn.Conv2d(192, 192, 1)

		self.conv9 = nn.Conv2d(192, 10, 1)

		self.avg = nn.AvgPool2d(6)


	def forward(self, x):
	    x = self.dp0(x)
	    x = self.conv1(x)
	    x = F.relu(x)
	    x = self.conv2(x)
	    x = F.relu(x)
	    x = self.conv3(x)
	    x = F.relu(x)
	    x = self.dp1(x)

	    x = self.conv4(x)
	    x = F.relu(x)
	    x = self.conv5(x)
	    x = F.relu(x)
	    x = self.conv6(x)
	    x = F.relu(x)
	    x = self.dp2(x)

	    x = self.conv7(x)
	    x = F.relu(x)
	    x = self.conv8(x)
	    x = F.relu(x)
	    x = self.conv9(x)
	    x = F.relu(x)
	    x = self.avg(x)
	    x = torch.squeeze(x)
	    return x

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print("Loss: {:.5f} Error: {:.3f}%".format(test_loss/len(test_loader), 100.*(1. - float(correct)/total)))
    return 100.*(1. - float(correct)/total)



__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                Increses dimension via padding, performs identity operations
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option="A"):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option=option)
        self.linear = nn.Linear(64, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(option):
    return ResNet(BasicBlock, [3, 3, 3], option=option)


def resnet32(option):
    return ResNet(BasicBlock, [5, 5, 5], option=option)


def resnet44(option):
    return ResNet(BasicBlock, [7, 7, 7], option=option)


def resnet56(option):
    return ResNet(BasicBlock, [9, 9, 9], option=option)


def resnet110(option):
    return ResNet(BasicBlock, [18, 18, 18], option=option)


def resnet1202(option):
    return ResNet(BasicBlock, [200, 200, 200], option=option)

