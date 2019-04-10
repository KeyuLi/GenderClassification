import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str, required=True, help='the path of photo you want to test')
parser.add_argument('output_path', type=str, required=True, help='the path of result you want to store')
args = parser.parse_args()

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
        self.conv = conv3x3(1, 16)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, 64, layers[0])
        self.layer2 = self.make_layer(block, 32, 128, layers[1],2)
        self.layer3 = self.make_layer(block, 64, 256, layers[2],2)
        self.layer4 = self.make_layer(block, 128, 512, layers[3],2)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(512*1*1, num_classes)

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

    def forward(self, x):
        out = self.conv(x)
        # print(out.shape)
        out = self.bn(out)
        out = self.leaky_relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        # print(out.shape)
        # out = self.max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


def Act():
    net = torch.load('net.pkl')
    face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')
    face_cascade.load('./cascades/haarcascade_frontalface_default.xml')
    frame = cv2.imread(args.input_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces :
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y: y+h, x: x+w]
            f = cv2.resize(roi_gray, (64, 64))
            f = f.reshape(1, 1, 64, 64)
            f = Variable(torch.cuda.FloatTensor(f))
            output= net(f)
            _, label = torch.max(output.data, 1)
            label = label.cpu().numpy()
            if (label == 0):
                cv2.putText(frame, 'Woman', (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, 255, 2)
            elif (label == 1):
                cv2.putText(frame, 'Man', (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, 255, 2)
            cv2.imwrite(args.output_path,frame)

if __name__ == '__main__':
    Act()