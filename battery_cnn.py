import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable

inputdata = np.load('inputdata.npy')
label = np.load('label.npy')
inputdata = np.load('inputdata.npy')
label = np.load('label.npy')
label = np.reshape(label, (28, 1))

BATCH_SIZE = 28
inputdata = torch.from_numpy(inputdata).float()
label = torch.from_numpy(label).float()
torch_dataset = Data.TensorDataset(inputdata, label)
trainloader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, )


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.layer_1 = nn.Conv2d(1, 16, kernel_size=(7, 7), stride=1, padding=3)
        self.mp = nn.MaxPool2d(2)
        self.layer_2 = nn.Conv2d(16, 16, kernel_size=(7, 7), stride=1, padding=3)
        self.layer_3 = nn.Conv2d(16, 16, kernel_size=(7, 7), stride=1, padding=3)
        self.layer_4 = nn.Linear(2592, 1)  # 全连接层

    def forward(self, x):
        x = x.view(x.size(0), 1, 20, 655)  # batch_size,channel,w,h
        # print(x.size())
        x = F.relu(self.mp(self.layer_1(x)))
        x = F.relu(self.mp(self.layer_2(x)))
        x = F.relu(self.mp(self.layer_3(x)))
        x = x.view(x.size(0), 2592)
        x = self.layer_4(x)
        return x


model = Actor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.MSELoss()
mean = []
for epoch in range(4000):
    loss_list = []
    for step, (batch_x, batch_y) in enumerate(trainloader, 0):
        prediction = model(batch_x)
        loss = criterion(prediction, batch_y)
        loss_list.append(loss.data.numpy())
        loss.backward()
        optimizer.step()
        # print(batch_y)
        if epoch == 3999:
            print('prediction:', prediction, 'batch_y: ', batch_y)
    mean.append(np.mean(loss_list))
    # print('epoch: ', epoch, 'mean loss: ', np.mean(loss_list))

plt.plot(prediction.data.numpy(), label='prediction')
plt.plot(batch_y.data.numpy(), label='label')
print(plt.legend())


