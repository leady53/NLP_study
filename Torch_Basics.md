# Torch basics

### 1. Training cycle
Imports
```
import torch
import torchvision  #for dataset
import torchvision.transforms as transforms
```
Data processing
```
#image to numpy
transform = transform.Compose(
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  

#load dataset, then put in DataLoader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
```
Build the net
```
import torch.nn as nn
import torch.nn.functional as F

def our_model(nn.Module):
    def __init__(self):
    #define the architecture
    def forward(self,x):
    #passing input x through each self.layer, add activation if need be
```
The Optimizers
```
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

```
##We can define a fit() function here

#epoch
    #mini-batch:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

```

Save and reload
```
torch.save(net.state_dict(), PATH)
net = Net()
net.load_state_dict(torch.load(PATH))
```

Testing
```
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
```







## 2. On the theory and Design

Q: why zero_grad()? 

A: Pytorch accumulates gradient. We want to accumulate the grad for each batch, then update the training in one shot (run each sample individually with the same model parameters and calculate the gradients without updating the model). The zero_grad() will restart aft each minib, then .backward() will destroy the DAG. 

Q: what is autograd?

A: the forward pass will define a computational graph (DAG); nodes = Tensors, edges = functions. Backpropagating through this graph then allows you to easily compute gradients. It is seen in two steps: (1) define forward (2) loss.backward(). But then we have nn.Module to take care of autograd already. 

Q: control flow & weight sharing in pytorch?

A: use for/if-else in definint forward() + reuse same parameters. 

Q: requires_grad? 

A: if it wants to be part of the DAG. Things like maxpool has no gradients. 

Q: Why use TensorBoard?

A: Show image and visualization of data, show DAG, Projector for embeddings/tensor viz, track loss, PR curve 