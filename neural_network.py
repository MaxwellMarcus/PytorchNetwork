import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tkinter import *

root = Tk()
canvas = Canvas(root,width=784,height=784)
canvas.pack()


train = datasets.MNIST('', train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST('', train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset = torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

net = Net()

optimizer = optim.Adam(net.parameters(),lr=0.001)

EPOCHS = 30

for epoch in range(EPOCHS):
    for data in trainset:
        X,y = data
        net.zero_grad()

        output = net(X.view(-1,784))

        loss = F.nll_loss(output,y)
        loss.backward()
        optimizer.step()

    print(loss)

correct = 0
total = 0
with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1,784))

        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print('Accuracy: ' + str(correct/total))



img = []
for i in range(784):
    img.append([0])

cd = False

mouse_pressed = False

def check_digit(event):
    global cd
    cd = True

def mouse_press(event):
    global mouse_pressed

    mouse_pressed = True

def mouse_release(event):
    global mouse_pressed
    mouse_pressed = False

def motion(event):
    global mouse_pressed
    global img

    if mouse_pressed:

        x = event.x//28
        y = event.y//28

        img[x+y*28] = [1]

root.bind('<Button-1>',mouse_press)
root.bind('<ButtonRelease-1>',mouse_release)
root.bind('<Motion>',motion)
root.bind('d',check_digit)


while True:
    if not cd:
        canvas.delete(ALL)
        for i in range(784):
            x = i % 28
            y = i // 28
            x *= 28
            y *= 28
            if img[i] == [1]:
                #print(x,y)
                canvas.create_rectangle(x,y,x+28,y+28,fill='black')

        root.update()
    else:
        data = torch.FloatTensor(img)
        print(torch.argmax(net(data.view(-1,784))))
        cd = False
        img = []
        for i in range(784):
            img.append([0])
