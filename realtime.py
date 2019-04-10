import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from mtcnn.src import detect_faces
from PIL import Image
import argparse

img_size = 178

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
def conv7x7(in_channels, out_channels, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7,
                     stride=stride, padding=1, bias=False)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*1*1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = self.pool(F.leaky_relu(self.conv5(x)))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x= self.fc3(x)
        return x

net = Net().cuda()
net.load_state_dict(torch.load('simpleNet2.pkl'))

def detect():
    videoCapture = cv2.VideoCapture(0)
    # fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # videoWriter = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
    success, img1 = videoCapture.read()
    img2 = Image.fromarray(img1)
    cv2.namedWindow('real-time')

    while success and cv2.waitKey(1) == -1:
        bounding_boxes, _ = detect_faces(img2)
        if len(bounding_boxes)!=0:
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            for i in range(0, len(bounding_boxes)):
                if bounding_boxes[i, 4] < 0.99:
                    continue
                x1, y1, x2, y2 = int(bounding_boxes[i, 0]), int(bounding_boxes[i, 1]), int(bounding_boxes[i, 2]), int(bounding_boxes[i, 3])
                img1 = cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
                roi_gray = gray[y1: y2, x1: x2]
                # print(bounding_boxes[i, 4])
                f = cv2.resize(roi_gray, (img_size, img_size))
                f = f.reshape(1, 1, img_size, img_size)
                f = Variable(torch.cuda.FloatTensor(f))
                output= net(f)
                _, label = torch.max(output.data, 1)
                label = label.cpu().numpy()
                if (label == 0):
                    cv2.putText(img1, 'Woman', (x1, y1 - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, 255, 2)
                elif (label == 1):
                    cv2.putText(img1, 'Man', (x1, y1 - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, 255, 2)
        cv2.imshow('real-time', img1)
        success, img1 = videoCapture.read()
        img2 = Image.fromarray(img1)
    videoCapture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()