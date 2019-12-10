#Made by Daniel Jambor, 12.03.2019
#
#Sources: pythonprogramming.net by harrison@pythonprogramming.net
#Pytorch.org
#stackoverflow.com

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Net import *



#if cuda is available, set it on and print the message, if not, set on cpu and print the message
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("\nRunning on the GPU\n")
else:
    device = torch.device("cpu")
    print("\nRunning on the CPU\n")


#creating our net
net = Net().to(device)


#loading our database
trainingData = np.load("./bin/trainingData.npy", allow_pickle=True)


#creating the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)
#creating the loss function
lossFunction = nn.MSELoss()


#loading our images to input and converting to the Tensor of (anyValue, imgSize) shape
input = torch.Tensor([i[0] for i in trainingData]).view(-1, 64, 64)
#divide by the number of pixels
input = input/255.0
#loading our matrix as Tensor - this is our correct answers
label = torch.Tensor([i[1] for i in trainingData])


#training function
def train(net):
    
    #setting the number of training examples utilized in one iteration
    BATCH_SIZE = 100
    #generation of our net
    EPOCHS = 89


    #in every generation
    for epoch in tqdm(range(EPOCHS)):
        
        #iterate over the length of trainset, taking steps of the size of batch size
        for i in range(0, len(input), BATCH_SIZE):
           
            #setting our batchInput and reshaping it
            batchInput = input[i:i+BATCH_SIZE].view(-1, 1, 64, 64).to(device)
            #setting our batchLabel 
            batchLabel = label[i:i+BATCH_SIZE].to(device)

            
            #setting the gradients to zero
            net.zero_grad()
            #getting our output
            output = net(batchInput)
            #compare output with batchLabel and setting the loss
            loss = lossFunction(output, batchLabel)
            #computes math stuff for every parameter which requires grad
            loss.backward()
            #performing a parameter update based on the current gradient
            optimizer.step()


#acrivating train function
train(net)

#saving trained net as model.pth
torch.save(net.state_dict(), './bin/model.pth')