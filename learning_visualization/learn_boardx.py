from tensorboardX import SummaryWriter
import torchvision
import torch.utils.data as Data
import torch
import torch.nn as nn

train_data = torchvision.datasets.MNIST(root="./data/", train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True)
test_data = torchvision.datasets.MNIST(root="./data/", transform=torchvision.transforms.ToTensor(),
                                       train=False, download=False)
test_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.mlp = nn.Sequential(
            nn.Linear(28 * 28, 10)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


mycnn = CNN()

writer = SummaryWriter('./results')
