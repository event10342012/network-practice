from torch import nn
from torch.nn import functional as F


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, inputs):
        return F.relu(self.conv(inputs))


class Model(nn.Module):
    def __init__(self, classes=1000):
        super(Model, self).__init__()
        # block 1
        self.conv1 = ConvNet(3, 64)
        self.conv2 = ConvNet(64, 64)

        # block 2
        self.conv3 = ConvNet(64, 128)
        self.conv4 = ConvNet(128, 128)

        # block 3
        self.conv5 = ConvNet(128, 256)
        self.conv5 = ConvNet(256, 256)
        self.conv5 = ConvNet(256, 256)
        self.conv5 = ConvNet(256, 256)

        # block 4
        self.conv5 = ConvNet(256, 512)
        self.conv5 = ConvNet(512, 512)
        self.conv5 = ConvNet(512, 512)
        self.conv5 = ConvNet(512, 512)

        # block 5
        self.conv5 = ConvNet(512, 512)
        self.conv5 = ConvNet(512, 512)
        self.conv5 = ConvNet(512, 512)
        self.conv5 = ConvNet(512, 512)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(3 * 3 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)

    def forward(self, inputs, training=True):
        # block 1
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # block 3
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv5(x)))
        x = self.pool(x)

        # block 4
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv5(x)))
        x = self.pool(x)

        # block 5
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv5(x)))
        x = self.pool(x)

        x = x.view(-1, 3 * 3 * 512)
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=training)
        x = F.dropout(F.relu(self.fc2(x)), 0.5, training=training)
        x = F.relu(self.fc3(x))
        return x
