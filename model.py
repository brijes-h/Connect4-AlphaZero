# ResNet structure
# layers of convolutions
# we will use image classification (here the 2d board is the image kind)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class C4Model(nn.Module):
	def __init__(self, device):
		super().__init__()
		self.device = device
		
		# defining the layers
        # 3 boards are passed 
        #  - one from player 1
        #  - another one that describes the empty spaces
        #  - one from the player 2
		# conv
		self.initial_conv = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=True)
		self.initial_bn = nn.BatchNorm2d(128)

		# Res block 1
		self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.res1_bn1 = nn.BatchNorm2d(128)
		self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.res1_bn2 = nn.BatchNorm2d(128)

		# Res block 2
		self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.res2_bn1 = nn.BatchNorm2d(128)
		self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.res2_bn2 = nn.BatchNorm2d(128)

		# value head -> outputs from -1(lose) to 1(win)
		self.value_conv = nn.Conv2d(128, 3, kernel_size=1, stride=1, bias=True)
		self.value_bn = nn.BatchNorm2d(3)
		self.value_fc = nn.Linear(3*6*7,32) # fully connected layer, for every value we need a neuron
		self.value_head = nn.Linear(32,1)

		# policy head -> this gives the best action/next move (from 0 to 6)
		self.policy_conv = nn.Conv2d(128, 32, kernel_size=1, stride=1, bias=True)
		self.policy_bn = nn.BatchNorm2d(32)
		self.policy_head = nn.Linear(32*6*7,7)
		self.policy_ls = nn.LogSoftmax(dim=1) # activation function layer (log softmax (used in AlphaGo))
        # this maps the array of outputs(7) between a value from 0 to 1 (probabilities of chosing an action)

	def forward(self,x):
		# define connections between the layers 