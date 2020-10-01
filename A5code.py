import numpy as np
import matplotlib.pyplot as plt 
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
To_Tensor = transforms.ToTensor()

mnist_trainset = datasets.MNIST(
	root='./data', train=True, download=True, transform=To_Tensor)
mnist_testset = datasets.MNIST(
	root='./data', train=False, download=True, transform=To_Tensor)
train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=128,shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_testset,batch_size=128, shuffle=True)

def  ws(input, W0, W1, b0, b1):
	return torch.matmul(nn.functional.relu(torch.matmul(input, W0.T)+b0), W1.T) +b1

def dn(input, W0, W1, W2, b0, b1, b2):
	temp = torch.matmul(input, W0.T)+b0
	return torch.matmul(nn.functional.relu(torch.matmul(nn.functional.relu(temp), W1.T) +b1), W2.T) +b2

def accuracyandloss(test_loader, net_type, W0=None, W1=None, W2=None, b0=None, b1=None, b2=None):
	a = 0
	loss = 0
	for i, j in tqdm(iter(test_loader)):
		i = torch.flatten(i, 1, 3)
		if net_type == 'wide':
			result = ws(i, W0, W1, b0, b1)
			pred = torch.argmax(result,1)
		elif net_type == 'deep':
			result = dn(i, W0, W1, W2, b0, b1, b2)
			pred = torch.argmax(result,1)

		a += torch.sum(pred == j)
		loss += torch.nn.functional.cross_entropy(result, j, size_average = False)
	
	return a.to(dtype=torch.float)/len(test_loader.dataset), loss/len(test_loader.dataset)


def trainws(dataload, LR, epochs, h , m):
	alpha = 1/np.sqrt(m)
	W0 = -2*alpha* torch.rand(h, m) + alpha
	W1 = -2*alpha* torch.rand(10, h) + alpha
	b0 = -2*alpha* torch.rand(h) + alpha
	b1 = -2*alpha* torch.rand(10) + alpha
	parameters = [W0, W1, b0, b1]
	opt = torch.optim.Adam(parameters, LR)
	loss_list = []
	for epoch in range(epochs):
		a = 0
		loss_list.append(0)
		for i, j in dataload:
			result = ws(i, W0, W1, b0, b1)
			pred = torch.argmax(result,1)
			a += torch.sum(pred == j)
			loss = torch.nn.functional.cross_entropy(result, j, size_average = False)
			opt.zero_grad()
			loss.backward()
			opt.step()
			loss_list[epoch] += loss
		loss_list[epoch] = loss_list[epoch]/len(dataload.dataset)
		a = a.to(dtype=torch.float)/len(dataload.dataset)
		if a>0.99:
			return loss_list, W0, W1, b0, b1
	return loss_list, W0, W1, b0, b1

loss_list_ws, W0_ws, W1_ws, b0_ws, b1_ws = trainws(train, 0.001, 500, 64, 784)

def traindn(dataload, LR, epochs, h , m):
    alpha = 1/np.sqrt(m)
    W0 = -2*alpha* torch.rand(h, m) + alpha
    W0.requires_grad = True
    W1 = -2*alpha* torch.rand(h, h) + alpha
    W1.requires_grad = True
    W2 = -2*alpha* torch.rand(10, h) + alpha
    W2.requires_grad = True
    b0 = -2*alpha* torch.rand(h) + alpha
    b0.requires_grad = True
    b1 = -2*alpha* torch.rand(h) + alpha
    b1.requires_grad = True
    b2 = -2*alpha* torch.rand(10) + alpha
    b2.requires_grad = True
    p = [W0, W1, W2, b0, b1, b2]
    opt = torch.optim.Adam(p, LR)
    loss_list = []
    for i in range(epochs):
        loss_list.append(0)
        a = 0
        for i, j in tqdm(iter(dataload)):
            i = torch.flatten(i, 1, 3)
            result = dn(i, W0, W1, W2, b0, b1, b2)
            pred = torch.argmax(result,1)
            a += torch.sum(pred == j)
            loss = torch.nn.functional.cross_entropy(result, j, size_average = False)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_list[i] += loss
            loss_list[i] = loss_list[i]/len(dataload.dataset)
            a = a.to(dtype=torch.float)/len(dataload.dataset)
    if a>0.99:
        return loss_list, W0, W1, W2, b0, b1, b2
    return loss_list, W0, W1, W2, b0, b1, b2


#a   
loss_list_ws, W0_ws, W1_ws, b0_ws, b1_ws = trainws(train, 0.001, 500, 64, 784)
print(loss_list_ws, W0_ws, W1_ws, b0_ws, b1_ws)
ws_acc, ws_loss = accuracyandloss(test, net_type ='wide', W0=W0_ws, W1=W1_ws, W2=None, b0=b0_ws, b1=b1_ws, b2=None)
print('accuracy for wide: ', ws_acc)
print('loss for wide: ', ws_loss)
plt.plot(range(len(loss_list_ws)), loss_list_ws, '-o', label = 'wide and shallow')
plt.xlabel('epoch')
plt.ylabel('loss')
#b
loss_list_dn, W0_dn, W1_dn, W2_dn, b0_dn, b1_dn, b2_dn = traindn(train, 0.001, 500, 32, 784)
dn_acc, dn_loss = accuracyandloss(test, net_type ='deep', W0=W0_dn, W1=W1_dn, W2=W2_dn, b0=b0_dn, b1=b1_ws, b2=b2_dn)
print('accuracy for wide: ', dn_acc)
print('loss for wide: ', dn_loss)
plt.plot(range(len(loss_list_dn)), loss_list_dn, '-o', label = 'deep and narrow')
plt.xlabel('epoch')
plt.ylabel('loss')
#c
numberws = np.prod(W0_ws.shape) + np.prod(W1_ws.shape) + np.prod(b0_ws.shape) + np.prod(b1_ws.shape)
numberdn = np.prod(W0_dn.shape) + np.prod(W1_dn.shape) + np.prod(W2_dn.shape) + np.prod(b0_dn.shape) + np.prod(b1_dn.shape) + np.prod(b2_dn.shape)
print(numberws, numberdn)
