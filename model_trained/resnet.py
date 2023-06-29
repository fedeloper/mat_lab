import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class ResNet_VAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=256, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_VAE, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)


        self.fc1 = nn.Linear(1024, self.fc_hidden1)

        self.bn1 = nn.BatchNorm1d(self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(2),
            nn.Conv2d(2,2,3),
            #nn.Conv2d(2, 2, 3),
            #nn.Conv2d(2, 2, 5),

            #nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )
        self.cx0= nn.Conv2d(1,3,3)
        self.cx1 = nn.Conv2d(1, 3, 3)
        self.last_con = nn.Conv2d(2,2,8)
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
        self.c = nn.Parameter(torch.ones(1))
    def encode(self, x0,x1):

        x0 = self.resnet(self.cx0(x0))  # ResNet
        x1 = self.resnet(self.cx1(x1))
        x0 = x0.view(x0.size(0), -1)  # flatten output of conv
        x1 = x1.view(x1.size(0), -1)

        x = torch.cat([x0,x1],-1)

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))

        return x


    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)

        x = F.interpolate(self.last_con(x), size=(400, 400), mode='bilinear')

        x = (x+( torch.stack(torch.meshgrid(torch.arange(400), torch.arange(400)), dim=-1).to("cuda").permute(2, 0, 1)/400*self.c)) * self.b + self.a

        return x #+grid

    def forward(self, x,y):
        emb = self.encode(x,y)


        x_reconst = self.decode(emb)#

        return x_reconst