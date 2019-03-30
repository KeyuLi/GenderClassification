from loadData import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchvision import transforms

transform = transforms.Compose(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

torch.cuda.set_device(0)

data_url1 = "./face_region/"
data_url2 = "./faceData/"
data_url3 = "./eyes_region/"
data_url4 = "./eyes_region/"


label_url = "./label.txt"
num = 15000
train_num = 10000
img_size = 178
channel = 1

X_train1, X_train2, X_train3, y_train, X_val1, X_val2, X_val3, y_val = load_data(data_url1, data_url2, data_url3, label_url, num, train_num, img_size, channel)


X_train1 = torch.FloatTensor(X_train1)
X_train2 = torch.FloatTensor(X_train2)
X_train3 = torch.FloatTensor(X_train3)
X_val1 = torch.FloatTensor(X_val1)
X_val2 = torch.FloatTensor(X_val2)
X_val3 = torch.FloatTensor(X_val3)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)

train_dataset1 = torch.utils.data.TensorDataset(X_train1, y_train)
train_dataset2 = torch.utils.data.TensorDataset(X_train2, y_train)
train_dataset3 = torch.utils.data.TensorDataset(X_train3, y_train)

train_data1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=128, shuffle=False, num_workers=0)
train_data2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=128, shuffle=False, num_workers=0)
train_data3 = torch.utils.data.DataLoader(dataset=train_dataset3, batch_size=128, shuffle=False, num_workers=0)
# train_data2 = torch.utils.data.DataLoader(train_dataset2, batch_size=128, shuffle=True, num_workers=0)
val_dataset1 = torch.utils.data.TensorDataset(X_val1, y_val)
val_dataset2 = torch.utils.data.TensorDataset(X_val2, y_val)
val_dataset3 = torch.utils.data.TensorDataset(X_val3, y_val)
val_data1 = torch.utils.data.DataLoader(val_dataset1, batch_size=16, shuffle=False, num_workers=0)
val_data2 = torch.utils.data.DataLoader(val_dataset1, batch_size=16, shuffle=False, num_workers=0)
val_data3 = torch.utils.data.DataLoader(val_dataset3, batch_size=16, shuffle=False, num_workers=0)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
def conv7x7(in_channels, out_channels, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7,
                     stride=stride, padding=1, bias=False)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, middle_channels, stride)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn. ReLU(inplace=True)
        self.conv2 = conv3x3(middle_channels, middle_channels)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = conv1x1(middle_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.leaky_relu(out)

        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.leaky_relu(out)
        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(channel, 16)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(True)
        self.leaky_relu = nn.LeakyReLU(True)
        self.drop_out = nn.Dropout2d(0.3)
        self.layer1 = self.make_layer(block, 16, 64, layers[0])
        self.layer2 = self.make_layer(block, 32, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 128, 512, layers[3], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(512*1*1, num_classes)
        # self.fc2 = nn.Linear(32, num_classes)
        # self.fc2 = nn.Linear(64, num_classes)
        # self.dropout = nn.Dropout(0.3)

    def make_layer(self, block, middle_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, middle_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, middle_channels, out_channels))
        return nn.Sequential(*layers)

    def forward1(self, data1, data2, data3):
        data1 = self.bn2(data1)
        data2 = self.bn2(data2)
        data3 = self.bn2(data3)

        data = data1 + data2 + data3
        out2 = self.conv(data)
        # out2 = self.conv(data2)
        # out3 = self.conv(data3)

        out2 = self.bn(out2)
        # out1 = self.bn(out1)
        # out3 = self.bn(out3)
        # print(out1)
        # out2 = self.bn(out2)
        # out1 += out2
        # out1 = self.max_pool(out1)

        out2 = self.relu(out2)
        # out1 = self.leaky_relu(out1)
        # out3 = self.leaky_relu(out3)
        # out2 = 5*out1 + 5*out3 + 2*out2
        # out2 = self.relu(out2).
        out2 = self.max_pool(out2)

        out2 = self.layer1(out2)
        out2 = self.layer2(out2)
        # out1 = self.max_pool(out1)
        out2 = self.layer3(out2)
        out2 = self.layer4(out2)
        # out2 = self.layer4(out2)
        # out1 += out2
        # out1 = self.avg_pool(out1)
        # out1 = out1.view(out1.size(0), -1)
        # out1 = self.fc1(out1)
        # out_pre = torch.cat((out1, out2), 1)
        # out2 = self.avg_pool(out2)
        out_pre = self.avg_pool(out2)
        # print(out_pre.size())
        # out2 = out2.view(out2.size(0), -1)
        out = out_pre.view(out_pre.size(0), -1)
        out = self.fc1(out)
        # out = self.drop_out(out)
        # out = self.fc2(out)
        return out

    def forward2(self, data):
        out = self.conv(data)
        # out = self.max_pool(out)
        # print(out.shape)
        out = self.bn(out)
        out = self.leaky_relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.max_pool(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out1 = self.avg_pool(out1)
        # out1 = out1.view(out1.size(0), -1)
        # out1 = self.fc1(out1)
        out = self.avg_pool(out)
        # out2 = out2.view(out2.size(0), -1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


# resnet = torch.load("net1.pkl").cuda()
layers = [3, 4, 6, 3]
# layers = [3, 4, 23, 3]
# layers = [3, 8, 36, 3]

resnet = ResNet(ResidualBlock, layers).cuda()
# resnet.load_state_dict(torch.load('resnet50_5.pkl'))

criterion = nn.CrossEntropyLoss()
optimzer = optim.Adam(resnet.parameters(), lr=1e-3)

def train(train_data1, train_data2, train_data3, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data1 in enumerate(train_data1, 0):
            _, data2 = list(enumerate(train_data2))[i]
            _, data3 = list(enumerate(train_data3))[i]

            inputs1, labels = data1
            inputs2 = data2[0]
            inputs3 = data3[0]
            inputs1, inputs2, labels = Variable(inputs1.cuda()),  Variable(inputs2.cuda()), Variable(labels.cuda())
            inputs3 = Variable(inputs3.cuda())
            outputs = resnet.forward1(inputs1, inputs2, inputs3)
            optimzer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimzer.step()
            running_loss += loss.data[0]
            if i % 100 == 0:
                print('[%d, %5d], loss: %.4f ' % (epoch+1, i+1, running_loss / 100))
                running_loss = 0.0


def test(val_data1, val_data2, val_data3):
    total = 0.0
    correct = 0.0
    for i, data1 in enumerate(val_data1, 0):
        _, data2 = list(enumerate(val_data2))[i]
        _, data3 = list(enumerate(val_data3))[i]
        inputs1, labels = data1
        inputs2 = data2[0]
        inputs3 = data3[0]
        inputs1 = Variable(inputs1.cuda())
        inputs2 = Variable(inputs2.cuda())
        inputs3 = Variable(inputs3.cuda())
        outputs = resnet.forward1(inputs1, inputs2, inputs3)
        _, labels_pre = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (labels_pre.cpu() == labels).sum()
    print('accuracy: %.2f %% ' % (100*correct / total))


if __name__ == '__main__':
    train(train_data1, train_data2, train_data3, 35)
    torch.save(resnet, 'combine.pkl')
    test(val_data1, val_data2, val_data3)
