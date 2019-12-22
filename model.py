import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1)
        self.cnn2 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2)
        self.cnn3 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3)
        self.cnn4 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4)
        self.line1 = nn.Linear(8,1)

        self.cnn21 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.cnn22 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn23 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, dilation=2)
        self.line2 = nn.Linear(4*64, 3)
        self.relu2 = nn.ReLU()

        self.cnna = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, dilation=2)
        self.bn = nn.BatchNorm3d(32)
        self.cnnb = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnnc = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3)
        self.line = nn.Linear(in_features=2688, out_features=60)


    def reset_parameters(self):
        self.cnn1.reset_parameters() 
        self.cnn2.reset_parameters() 
        self.cnn3.reset_parameters() 
        self.cnn4.reset_parameters() 
        self.line1.reset_parameters()

        self.cnn21.reset_parameters()
        self.cnn22.reset_parameters()
        self.cnn23.reset_parameters()
        self.line2.reset_parameters()

    def forward(self, x):
        # x = x.permute(0,3,1,2)
        x = self.cnna(x.permute(0,3,1,2))
        # print(x.size())
        x1 = self.relu2(x)
        # print(x1.size())
        x2 = self.relu2(self.cnnb(x1))
        # print(x2.size())
        x3 = self.relu2(self.cnnc(x2))
        # print(x3.size())
        x3 = x3.view(x.size(0), -1)
        # print(x3.size())
        x3 = self.line(x3)
        return 0, x3
        # print(x2.size())
    #     x1 = self.cnn1(x[3:-1,...])
    #     x2 = self.cnn2(x[2:-1,...])
    #     x3 = self.cnn3(x[1:-1,...])
    #     x4 = self.cnn4(x[:-1,...])
    #     # print(x1.size(), x2.size(), x4.size())
    #     x11=torch.zeros((x1.size(0), 1, x1.size(2))).to('cuda:1')
    #     x12=torch.zeros((x1.size(0), 1, x1.size(2))).to('cuda:1')
    #     x13=torch.zeros((x1.size(0), 1, x1.size(2))).to('cuda:1')
    #     x14=torch.zeros((x1.size(0), 1, x1.size(2))).to('cuda:1')
    #     for i in range(x1.size(2)):
    #         x11[..., i] = self.line1(x1[..., i])
    #         x12[..., i] = self.line1(x2[..., i])
    #         x13[..., i] = self.line1(x3[..., i])
    #         x14[..., i] = self.line1(x4[..., i])
    #     # print(x11.size())
    #     out1 = torch.cat((x11,x12,x13,x14), dim=1)
    #     # print(out1.size(), x[..., 4:].size())
    #     x20 = torch.zeros((out1.size(0), 1, out1.size(1)+1, out1.size(2))).to('cuda:1')
    #     x20[:, 0, ...] = torch.cat((x[..., 4:], out1), dim=1)
    #     x21 = self.relu2(self.cnn21(x20))
    #     x22 = self.relu2(self.cnn22(x21))
    #     x23 = self.relu2(self.cnn23(x22[...,0,:]))
    #     out2 = self.line2(x23.reshape([x23.size(0), -1]))
        
        # print(x20.size(), x21.size(), x22.size(), x23.size())
        # return out1, out2 
        a =  torch.zeros((50, 60)).to('cuda:1')
        b = torch.zeros((50, 60)).to('cuda:1')
        b = torch.randn((50,60)).to('cuda:1')
        return a, x2

