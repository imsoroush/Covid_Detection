from matplotlib.pyplot import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import torch
import torch.optim as optim
import time
import psutil
import glob
import os
import random


# Label
labels = [1 for _ in covid] + [0 for _ in noncovid]
labels = np.array(labels)

# RGB2Gray
x = []
for img in image_path:
  img_data = rgb2gray(imread(img))
  new_img = resize(img_data, (256, 256))
  x.append(new_img)

# reshape and Shuffle
x = np.array(x).reshape((len(x), 1, 256, 256)) 
np.random.seed(123)
np.random.shuffle(x)
np.random.seed(123)
np.random.shuffle(labels)

# CNN Model
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.group1_conv = torch.nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2)
        self.group1_bn = torch.nn.BatchNorm2d(4)
        self.group1_tanh = torch.nn.Tanh()
        self.group1_avgpool = torch.nn.AvgPool2d(5, stride=2, padding=2)

        torch.nn.init.normal_(self.group1_conv.weight, 0, 0.01)

        self.group2 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, kernel_size=5, bias=True, stride=1, padding=2),
            torch.nn.BatchNorm2d(8),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(5, stride=2, padding=2)
        )
        torch.nn.init.normal_(self.group2[0].weight, 0, 0.01)

        self.group3 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=5, bias=True, stride=1, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(5, stride=2, padding=2),
        )
        torch.nn.init.normal_(self.group3[0].weight, 0, 0.01)

        self.group4 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, bias=True, stride=1, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(5, stride=2, padding=2),
        )
        torch.nn.init.normal_(self.group4[0].weight, 0, 0.01)

        self.group5 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, bias=True, stride=1, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(5, stride=2, padding=2),
        )
        torch.nn.init.normal_(self.group5[0].weight, 0, 0.01)

        self.group6 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=5, bias=True, stride=1, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(5, stride=2, padding=2),
        )
        torch.nn.init.normal_(self.group6[0].weight, 0, 0.01)

        self.group7 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=5, bias=True, stride=1, padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(5, stride=2, padding=2),
        )
        torch.nn.init.normal_(self.group7[0].weight, 0, 0.01)

        self.group8 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=5, bias=True, stride=1, padding=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(5, stride=2, padding=2),
        )
        torch.nn.init.normal_(self.group8[0].weight, 0, 0.01)

        self.group9 = torch.nn.Sequential(
            torch.nn.Linear(1024, 100),
            torch.nn.Linear(100, 1),

        )
        torch.nn.init.normal_(self.group9[0].weight, 0, 0.01)
        torch.nn.init.normal_(self.group9[1].weight, 0, 0.01)



    def forward(self, x):
        x1 = self.group1_conv(x)
        x1 = torch.abs(x1)
        x1 = self.group1_bn(x1)
        x1 = self.group1_tanh(x1)
        x1 = self.group1_avgpool(x1)
        # print(x1.size())
        x2 = self.group2(x1)
        # print(x2.size())
        x3 = self.group3(x2)
        # print(x3.size())
        x3 = self.group4(x3)
        # print(x3.size())
        x3 = self.group5(x3)
        # print(x3.size())
        x3 = self.group6(x3)
        # print(x3.size())
        x3 = self.group7(x3)
        # print(x3.size())
        #x3 = self.group8(x3)
        #print(x3.size())
        (_, C, H, W) = x3.data.size()
        x3 = x3.view(-1, C * H * W)
        # print(x3.size())
        x3 = self.group9(x3)
        # print(x3.size())
        x3 = torch.nn.functional.sigmoid(x3)
        return x3

# Train Model
def trainNet(net, batch_size, n_epochs, learning_rate, step_lr, lr_decay,
             train_data, train_label, save_path, save_every_n_epoch, valid_acc_thr,
             is_cuda, verbose=True):
    cuda_cpu = 'cpu'
    if is_cuda:
        cuda_cpu = 'cuda'
        net.cuda()
    # Create our loss and optimizer functions
    loss = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    sched = optim.lr_scheduler.StepLR(optimizer, step_lr, lr_decay)

    # Time for printing
    x_train = np.copy(train_data[:1600])
    x_valid = np.copy(train_data[1600:])

    y_train = np.copy(train_label[:1600]).reshape((1600, 1))
    y_valid = np.copy(train_label[1600:]).reshape((399, 1))

    num_batch_train = x_train.shape[0] // batch_size
    num_batch_valid = x_valid.shape[0] // batch_size

    best_valid_acc = 0
    best_valid_epoch = 0

    print('=' * 23 + 'Model' + '=' * 22)
    print(net)
    print('=' * 18 + 'Start Training' + '=' * 18)
    print(f'num_batch_train: {num_batch_train}')

    training_start_time = time.time()
    # Loop for n_epochs
    for epoch in range(n_epochs):
        start_time_epoch = time.time()
        total_train_loss = 0
        correct_epoch = 0
        for batch in range(num_batch_train):
            x_batch = torch.from_numpy(x_train[batch*batch_size:(batch+1)*batch_size]).float().to(cuda_cpu)
            y_batch = torch.from_numpy(y_train[batch*batch_size:(batch+1)*batch_size]).float().to(cuda_cpu)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            out = net(x_batch)
            batch_loss = loss(out, y_batch)
            batch_loss.backward()
            optimizer.step()
            sched.step()
            out[out >= 0.5] = 1
            out[out < 0.5] = 0

            correct_batch = (out == y_batch).sum().item()
            correct_epoch += correct_batch

            acc = 100.0 * correct_batch / batch_size

            # Print statistics
            total_train_loss += batch_loss.data
            if verbose:
                print('train:[%d],step:%d,loss:%f, acc:%f' % (epoch, batch, batch_loss, acc))

        valid_loss = 0
        correct_valid = 0
        with torch.no_grad():
            for batch in range(num_batch_valid):
                x_batch = torch.from_numpy(x_valid[batch*batch_size:(batch+1)*batch_size]).float().to(cuda_cpu)
                y_batch = torch.from_numpy(y_valid[batch*batch_size:(batch+1)*batch_size]).float().to(cuda_cpu)

                # Forward pass
                out = net(x_batch)
                valid_loss += loss(out, y_batch)

                out[out >= 0.5] = 1
                out[out < 0.5] = 0

                correct_valid += (out == y_batch).sum().item()
        print('=' * 50)
        train_loss = total_train_loss / num_batch_train
        train_acc = 100.0 * correct_epoch / (batch_size * num_batch_train)
        valid_loss = valid_loss / num_batch_valid
        valid_acc = 100.0 * correct_valid / (batch_size * num_batch_valid)
        end_time_epoch = time.time()
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_epoch = epoch
            if valid_acc >= valid_acc_thr:
                pass
        print('epoch[%d] | train:: loss:%f, acc:%f | valid:: loss: %f, acc: %f | ETA: %f minutes | best valid acc: %f, '
              'best valid acc epoch: %d | lr: %f' % (epoch, train_loss, train_acc, valid_loss, valid_acc,
                                                     (end_time_epoch - start_time_epoch) / 60, best_valid_acc,
                                                     best_valid_epoch, optimizer.param_groups[0]['lr']))
        print('=' * 50)
        if (epoch+1) % save_every_n_epoch == 0:
          pass
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return net

net = trainNet(ConvNet(),
             32,
             20,
             0.001,
             100,
             0.9,
             x,
             labels,
             "",
             20,
             65,
             True,
             False)

