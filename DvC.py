from os import listdir
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

img_size = 64
labels = {"cat": 0, "dog": 1}
name_td = "dvc_td.npy"
name_model = "dvc_CNN.pt"
l_rate = 0.001
b_size = 100
EPOCHS = 5

class CvD():
    files = listdir("./train")
    training_data = []
    n = np.zeros(3, dtype=int)

    def build_td(self, BLD_DATA):
        if BLD_DATA == True:
            for name in tqdm(self.files):
                try:
                    lbl = np.eye(2)[labels.get(str(name.split(".")[0]))]
                    self.n[labels.get(str(name.split(".")[0]))] += 1
                    img = cv2.resize(cv2.imread(f"./train/{name}", cv2.IMREAD_GRAYSCALE), (img_size, img_size)) / 255
                    self.training_data.append([np.array(img), lbl])
                    # print(lbl, img)
                    # plt.imshow(img, cmap="gray")
                    # plt.show()
                except Exception as e:
                    self.n[2] += 1
                    pass
            np.random.shuffle(self.training_data)
            np.save(name_td, self.training_data)
            print("Data distribution:", self.n)
        else: print("Build mode off")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)

        x = torch.randn(img_size, img_size).view(-1,1,img_size,img_size)
        self.linear_len = None
        self.convs(x)

        self.fc1 = nn.Linear(self.linear_len, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.linear_len is None:
            # self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            self.linear_len = len(torch.flatten(x))
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.linear_len)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = CNN().to(device)
# print(net)

class runCNN():
    dataset = np.load(name_td, allow_pickle=True)
    trainPercentage = 0.9
    trainSize = int(trainPercentage * len(dataset))
    X = torch.Tensor([i[0] for i in dataset]).view(-1, img_size, img_size)
    y = torch.Tensor([i[1] for i in dataset])
    train_X = X[:trainSize]
    train_y = y[:trainSize]
    test_X = X[trainSize:]
    test_y = y[trainSize:]

    def train(self, net):
        decayRate = 0.9
        optimizer = optim.Adam(net.parameters(), lr=l_rate)
        loss_function = nn.MSELoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
        for epoch in range(EPOCHS):
            for i in range(0, len(self.train_X), b_size):
                batch_X = self.train_X[i:i + b_size].view(-1, 1, img_size, img_size).to(device)
                batch_y = self.train_y[i:i + b_size].to(device)
                outputs = net(batch_X)
                # print(batch_y, outputs)
                net.zero_grad()
                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()
                if i % b_size*2 == 0:
                    print(f"Epoch: {epoch + 1} / {EPOCHS} | {round(100* i / len(self.train_X), 3)} %\t| Loss: {loss}")
            lr_scheduler.step()
        torch.save(net.state_dict(), name_model)

    def test(self, net):
        net.load_state_dict(torch.load(name_model))
        net.eval()
        correct_classes = np.zeros(len(labels), dtype=float)
        total_classes = np.zeros(len(labels), dtype=float)
        with torch.no_grad():
            print(f"Testing...")
            for i in range(len(self.test_X)):
                true_class = torch.argmax(self.test_y[i].to(device))
                # print(np.shape(self.test_X[i].view(-1, 1, img_size, img_size)), np.dtype(self.test_X[i].view(-1, 1, img_size, img_size)))
                outputs = net(self.test_X[i].view(-1, 1, img_size, img_size).to(device))[0]
                predicted_class = torch.argmax(outputs)
                if true_class == predicted_class:
                    correct_classes[true_class] += 1
                total_classes[true_class] += 1
        correct = np.sum(correct_classes)
        total = np.sum(total_classes)
        acc_classes = np.round(correct_classes / total_classes, 3)
        print(f"Overall accuracy: {round(100 * correct / total, 3)}%")
        for i, key in enumerate(labels.keys()):
            print(f"{key}: {100 * acc_classes[i]}%")

    def testimg(self, net):
        net.load_state_dict(torch.load(name_model))
        net.eval()
        net = net.float()
        with torch.no_grad():
            print(f"Testing...")
            file = listdir("./figused")[0]
            img = np.array(cv2.resize(cv2.imread(f"./figused/{file}", cv2.IMREAD_GRAYSCALE), (img_size, img_size)) / 255)
            outputs = net(torch.tensor([[img]]).float())[0]
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.title(f"CAT: {round(100*float(outputs[0]), 3)}%\nDOG: {round(100*float(outputs[1]), 3)}%",
                  fontsize="large", weight="bold")
        plt.show()

# a = CvD()
# a.build_td(BLD_DATA=False)
a = runCNN()
# a.train(net)
a.test(net)
# a.testimg(net)