import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size = 5,stride = 1,padding = 2)#padding = 2,so the size of the image won't change
        self.conv2 = nn.Conv2d(10,20,kernel_size = 5, stride = 1,padding = 2)
        self.mp = nn.MaxPool2d(kernel_size = 2)
        self.fc = nn.Linear(980,10)

    def forward(self, x):
        in_size = x.size(0)
        out = F.relu(self.mp(self.conv1(x)))
        #
        out = F.relu(self.mp(self.conv2(out)))
        out = out.view(in_size,-1)
        out = self.fc(out)
        # print(out.size())
        return F.log_softmax(out)




class Main():
    def __init__(self):
        self.net = torch.load("./net.tar")
        # print(self.net)
        self.predict("1.png")



    def predict(self,path):
        img = cv2.imread(path,0)
        img = torch.tensor(img[np.newaxis,np.newaxis,:,:]).type('torch.FloatTensor').cuda()
        output = self.net(img)
        pred = output.data.max(1, keepdim=True)[1]
        print(pred.item())






if __name__ == "__main__":
    t = Main()
