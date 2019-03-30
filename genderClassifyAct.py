import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2 )
        self.conv4 = nn.Conv2d(64, 128, 5, stride=1, padding=1 )
        self.conv5 = nn.Conv2d(128, 256, 5, stride=1, padding=2 )

        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(256*4*4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 256*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x= self.fc4(x)
        return x

net = torch.load('net.pkl').cuda()

def detect():
    face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')
    face_cascade.load('./cascades/haarcascade_frontalface_default.xml')
    videoCapture = cv2.VideoCapture('video01.mkv')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('video01_out.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    success, frame = videoCapture.read()

    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces :
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y: y+h, x: x+w]
            f = cv2.resize(roi_gray, (178, 178))
            f = f.reshape(1, 1, 178, 178)
            f = Variable(torch.cuda.FloatTensor(f))
            output= net(f)
            _, label = torch.max(output.data, 1)
            label = label.cpu().numpy()
            if (label == 0):
                cv2.putText(frame, 'Woman', (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, 255, 2)
            elif (label == 1):
                cv2.putText(frame, 'Man', (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, 255, 2)
        videoWriter.write(frame)
        success, frame = videoCapture.read()
        # cv2.imshow("genderClassify", frame)
        # if (cv2.waitKey(1) & 0xff == ord("q")):
        #     break

if __name__ == '__main__':
    detect()