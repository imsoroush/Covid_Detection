from matplotlib.pyplot import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import random
import torch 
from torch import nn
import torch.optim as optim
import time
import psutil
import glob
import os

# Label
labels_test_org = [1 for _ in covid] + [0 for _ in noncovid]
labels_test_org = np.array(labels_test_org).reshape((len(labels_test_org), 1))

# reshape and RGB2Gray
x_test_org = []
for img in image_path:
  img_data = rgb2gray(imread(img))
  new_img = resize(img_data, (256, 256))
  x_test_org.append(new_img)

x_test_org = np.array(x_test_org).reshape((len(x_test_org), 1, 256, 256)) 
x_test_org /= 255

# Model
batch_size = 1
num_batch_test = x_test.shape[0] // batch_size
net.eval()
cuda_cpu = 'cuda'
loss = torch.nn.BCELoss()
test_loss = 0
correct_test = 0
with torch.no_grad():
    for batch in range(num_batch_test):
        x_batch = torch.from_numpy(x_test[batch*batch_size:(batch+1)*batch_size]).float().to(cuda_cpu)
        y_batch = torch.from_numpy(labels_test[batch*batch_size:(batch+1)*batch_size]).float().to(cuda_cpu)

        # Forward pass
        out = net(x_batch)
        test_loss += loss(out, y_batch)
        
        # out = outputs.to('cpu')
        out[out >= 0.5] = 1
        out[out < 0.5] = 0

        correct_test += (out == y_batch).sum().item()
        # print(correct_batch, out.size())
print('=' * 50)
test_loss = test_loss / num_batch_test
test_acc = 100.0 * correct_test / (batch_size * num_batch_test)
print(f'test_loss: {test_loss}, test_acc: {test_acc}')

# GAN Model
class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, 1)
        # softmax and sigmoid
        self.softmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        # print(conv1.size())
        conv2 = self.conv2(conv1)
        # print(conv2.size())
        conv3 = self.conv3(conv2)
        # print(conv3.size())
        conv4 = self.conv4(conv3)
        # print(conv4.size())
        conv5 = self.conv5(conv4)
        # print(conv5.size())
        conv6 = self.conv6(conv5)
        # print(conv6.size())
        flat6 = conv6.view(-1, 4*4*512)
        # print(flat6.size())
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)
        classes = self.sigmoid(fc_aux)
        realfake = self.sigmoid(fc_dis)
        return realfake, classes

class _netG(nn.Module):
    def __init__(self, nz):
        super(_netG, self).__init__()
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(100, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv7 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 8, 2, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        input = input.view(-1, self.nz)
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 768, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        tconv5 = self.tconv6(tconv5)
        tconv5 = self.tconv7(tconv5)
        output = tconv5
        return output

gnet = _netG(100)
dnet = _netD()
x__ = np.zeros((2, 100, 1, 1))
x_batch = torch.from_numpy(x__).float()

a = gnet(x_batch)
b = dnet(a)

# Train Model
def trainNet(netG, netD, batch_size, n_epochs, learning_rate, step_lr, lr_decay,
             train_data, train_label, save_path, save_every_n_epoch, valid_acc_thr,
             is_cuda, verbose=True):
    cuda_cpu = 'cpu'
    if is_cuda:
        cuda_cpu = 'cuda'
        netG.cuda()
        netD.cuda()
    # Create our loss and optimizer functions
    dis_loss = torch.nn.BCELoss()
    # aux_loss = torch.nn.NLLLoss()
    aux_loss = torch.nn.BCELoss()

    dis_optimizer = optim.Adam(netD.parameters(), lr=learning_rate)
    gen_optimizer = optim.Adam(netG.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    dis_sched = optim.lr_scheduler.StepLR(dis_optimizer, step_lr, lr_decay)
    gen_sched = optim.lr_scheduler.StepLR(gen_optimizer, step_lr, lr_decay)

    # Time for printing
    x_train = np.copy(train_data)
    y_train = np.copy(train_label).reshape((1999, 1))
    num_batch_train = x_train.shape[0] // batch_size
    # num_batch_valid = x_valid.shape[0] // batch_size
    nz = 100
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
    y_fake_batch = torch.from_numpy(np.zeros((batch_size, 1))).float().to(cuda_cpu)
    y_real_batch = torch.from_numpy(np.ones((batch_size, 1))).float().to(cuda_cpu)

    print('=' * 23 + 'Model' + '=' * 22)
    print(netG)
    print('=' * 18 + 'Start Training' + '=' * 18)
    print(f'num_batch_train: {num_batch_train}')

    training_start_time = time.time()
    # Loop for n_epochs
    for epoch in range(n_epochs):
        start_time_epoch = time.time()
        total_train_loss_d = 0
        total_train_loss_g = 0
        correct_epoch = 0
        for batch in range(num_batch_train):
            correct = 0
            x_batch = torch.from_numpy(x_train[batch*batch_size:(batch+1)*batch_size]).float().to(cuda_cpu)
            y_batch = torch.from_numpy(y_train[batch*batch_size:(batch+1)*batch_size]).float().to(cuda_cpu)
            
            # train with real
            netD.zero_grad()
            
            s_output, c_output = netD(x_batch)
            s_errD_real = dis_loss(s_output, y_real_batch)
            c_errD_real = aux_loss(c_output, y_batch)
            errD_real = s_errD_real + c_errD_real
            errD_real.backward()
            #
            c_output[c_output >= 0.5] = 1 
            c_output[c_output < 0.5] = 0
            correct = (c_output == y_batch).sum().item()
            correct_epoch += correct
            

            # train with fake
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)

            label = np.random.randint(0, 2, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            noise_[:, 0] = label
            
            noise_ = (torch.from_numpy(noise_))
            noise_ = noise_.resize_(batch_size, nz, 1, 1)
            noise.data.copy_(noise_)

            y_batch = torch.from_numpy(label.reshape((batch_size, 1))).float().to(cuda_cpu)

            fake = netG(noise.to(cuda_cpu))
            
            s_output,c_output = netD(fake.detach())
            s_errD_fake = dis_loss(s_output, y_fake_batch)
            c_errD_fake = aux_loss(c_output, y_batch)
            errD_fake = s_errD_fake + c_errD_fake

            errD_fake.backward()
            D_G_z1 = s_output.data.mean()
            errD = errD_fake + errD_real
            s_errD = s_errD_fake + s_errD_real
            c_errD = c_errD_fake + c_errD_real
            dis_optimizer.step()
            dis_sched.step()

            total_train_loss_d += errD

            netG.zero_grad()
            fake = netG(noise.to(cuda_cpu))
            s_output,c_output = netD(fake)
            s_errG = dis_loss(s_output, y_real_batch)
            c_errG = aux_loss(c_output, y_batch)
            
            errG = s_errG + c_errG
            errG.backward()
            D_G_z2 = s_output.data.mean()
            gen_optimizer.step()

            # second time train
            errG2 = 0
            for _ in range(2):
                netG.zero_grad()
                noise.data.resize_(batch_size, nz, 1, 1)
                noise.data.normal_(0, 1)

                label = np.random.randint(0, 2, batch_size)
                noise_ = np.random.normal(0, 1, (batch_size, nz))
                noise_[:, 0] = label
                
                noise_ = (torch.from_numpy(noise_))
                noise_ = noise_.resize_(batch_size, nz, 1, 1)
                noise.data.copy_(noise_)

                fake = netG(noise.to(cuda_cpu))
                s_output,c_output = netD(fake)
                s_errG = dis_loss(s_output, y_real_batch)
                c_errG = aux_loss(c_output, y_batch)
                
                errG_tmp = s_errG + c_errG
                errG_tmp.backward()
                errG2 += errG_tmp 
                D_G_z2 = s_output.data.mean()
                gen_optimizer.step()
            gen_sched.step()
            
            total_train_loss_g += (errG + errG2)/3

            if verbose:
                print('train:[%d],step:%d,errD:%f,s_errD:%f, c_errD:%f, errG:%f,s_errG:%f,c_errG:%f, acc:%f' % (epoch, batch, errD, s_errD, c_errD, errG, s_errG, c_errG,  100.*correct/batch_size))
        
        end_time_epoch = time.time()
        print('=' * 50)
        print('epoch[%d] | train:: loss_d:%f, loss_g: %f, acc:%f | ETA: %f minutes | lr: %f' % (epoch, total_train_loss_d/num_batch_train, total_train_loss_g/num_batch_train,100.*correct_epoch/(batch_size*num_batch_train),
                                                     (end_time_epoch - start_time_epoch) / 60, dis_optimizer.param_groups[0]['lr']))
        print('=' * 50)
        if epoch % 10 == 1:
          torch.save(netD.state_dict(), save_path + '/netD_' + str(epoch) + '.state')
          torch.save(netG.state_dict(), save_path + '/netG_' + str(epoch) + '.state')
          torch.save(netD, save_path + '/netD_' + str(epoch) + ".model")
          torch.save(netG, save_path + '/netG_' + str(epoch) + ".model")
    torch.save(netD.state_dict(), save_path + '/netD.state')
    torch.save(netG.state_dict(), save_path + '/netG.state')
    torch.save(netD, save_path + '/netD.model')
    torch.save(netG, save_path + '/clf_xu_pytorch_final.model')
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return netG, netD

# netD = NetD()
netD = _netD()
# netG = NetG(100, 32, 1)
netG = _netG(100)

netG, netD = trainNet(netG, netD,
                      32,
                      500,
                      0.0001,
                      2220,
                      0.9,
                      x,
                      labels,
                      "/content/drive/My Drive/models",
                      20,
                      65,
                      True,
                      True)

