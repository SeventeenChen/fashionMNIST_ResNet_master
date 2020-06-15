# -*- coding: utf-8 -*-

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

BATH_SIZE = 128

# Note transforms.ToTensor() scales input images
# to 0-1 range


train_dataset = datasets.FashionMNIST(root='data',
									  train=True,
									  transform= transforms.ToTensor(),
									  download=False)

test_dataset = datasets.FashionMNIST(root='data',
									 train=False,
									 transform= transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
						  batch_size=BATH_SIZE,
						  shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
						 batch_size=BATH_SIZE,
						 shuffle=True)

# # Checking the dataset
# for images, labels in train_loader:
# 	print('Image batch dimensions:', images.shape)
# 	print('Image label dimensions:', labels.shape)
# 	break
