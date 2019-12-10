#Made by Daniel Jambor, 12.03.2019
#
#Sources: pythonprogramming.net by harrison@pythonprogramming.net
#Pytorch.org
#stackoverflow.com

import torch
import torch.nn as nn
import torch.nn.functional as F

#creating class inhereits from nn.Module - the base class for all neural network modules
class Net(nn.Module):

    #init function
    def __init__(self):
        
        #super() - inhereits from nn.Module and run Net's init method
        super().__init__()
        
        #creating convolutional layers; nn.Conv2d - applies a 2D convolution over an input signal composed of several input planes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

        #creating some fake data to check the shape of the flattened output
        x = torch.randn(64, 64).view(-1, 1, 64, 64)
        self.toLinear = None
        self.convs(x)

        #creating fully connected layers; nn.Linear(size of each input, size of each output) - applies a linear transformation to the incoming data
        self.fc1 = nn.Linear(in_features=self.toLinear, out_features=1024) #flattering
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    #function which passes data through the conv layers
    def convs(self, x):
        
        #passing through activation function: F.max_pool2d - applies a 2D max pooling over an input signal composed of several input planes (here: 2x2), then runs activation function
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2)

        #checking shape to flatting
        if self.toLinear is None:
            self.toLinear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x
    
    #fuction which passes data through the fc layers
    def forward(self, x):
        
        x = self.convs(x)

        #view - returns a new tensor with the same data but of different shape; -1 means any value
        x = x.view(-1, self.toLinear)
        
        #passing through activation function F.relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        #last layer is output - dont run activation function
        x = self.fc3(x)

        #return softmax of x in dimension 1
        return F.softmax(x, dim=1)