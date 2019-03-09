import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as Transform
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy as np



ROOT = "./data/"
TRAIN_PATH = "./data/mnist/train/train.txt"
TEST_PATH = "./data/mnist/test/test.txt"
BATCH_SIZE = 64



class MyDataset(Dataset):
    def __init__(self,txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_lists = [ROOT + i.split(" ")[0] for i in lines]

            self.label_lists = [int(i.split(" ")[1][0]) for i in lines]
    def __getitem__(self, index):
        img = cv2.imread(self.img_lists[index],0)
        img = img[np.newaxis,:,:]#add dimention
        label = torch.LongTensor([self.label_lists[index]])
        return img,label

    def __len__(self):
        return len(self.label_lists)



# print(train_loader)


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
    def train(self,epoch):
        train_dataset = MyDataset(TRAIN_PATH)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)



        for step,(data,target) in enumerate(train_loader):
            data = data.type('torch.FloatTensor').cuda()
            target = target.squeeze().cuda()
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if step % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset), 100. * step / len(train_loader),
                loss.item()))
    def test(self):
        test_loss = 0
        correct = 0
        test_dataset = MyDataset(TEST_PATH)
        test_loader = DataLoader(dataset = test_dataset,batch_size = BATCH_SIZE,shuffle = False)
        for data,target in test_loader:
            data = data.type('torch.FloatTensor').cuda()
            target = target.squeeze().cuda()
            output = self.net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()# get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


    def main(self):
        self.net = Net().cuda()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.5)#very important,if it is too big,the training process doesn't work!!!
        for epoch in range(10):
            self.train(epoch)
            self.test()


        torch.save(self.net,"./net.tar")


# for batch_idx, (data, target) in enumerate(train_loader):
#     print(data)
#     print(target)

if __name__ == "__main__":
    t = Main()
    t.main()





