from loadData import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

torch.cuda.set_device(0)

data_url = "./eyenose/"
data_url1 = "./face_region/"
data_url2 = "./eyenose_region/"

label_url = "./label.txt"
num = 65000
train_num = 60000
img_size = 178
channel = 1


X_train, y_train, X_val, y_val = load_data(data_url, label_url, num, train_num, img_size, channel)

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_data = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
val_data = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(channel, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*3*3, 64)
        # self.fc2 = nn.Linear(64, 64)
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
        # x = F.leaky_relu(self.fc2(x))
        x= self.fc3(x)
        return x

net = Net().cuda()
# net.load_state_dict(torch.load('simpleNet9.pkl'))
criterion = nn.CrossEntropyLoss()
optimzer = optim.Adam(net.parameters(), lr=1e-3)

def train(train_data, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net.forward(inputs)
            optimzer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimzer.step()
            running_loss += loss.data[0]
            if i % 100 == 0 :
                print('[%d, %5d], loss: %.4f ' % (epoch+1, i+1, running_loss / 100))
                running_loss = 0.0

def test(val_data):
    total = 0.0
    correct = 0.0
    for data in val_data:
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        outputs = net.forward(inputs)
        _, labels_pre = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (labels_pre.cpu() == labels).sum()
    print('accuracy: %.2f %% ' % (100*correct / total))


if __name__ == '__main__':
    train(train_data, 45)
    #60
    # torch.save(net.state_dict(), 'simpleNet16.pkl')
    test(val_data)