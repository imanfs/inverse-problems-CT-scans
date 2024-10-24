import torch.nn as nn
import torch 

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.linearproj = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        else:
            self.linearproj = None

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.linearproj is not None:
            identity = self.linearproj(identity)
        out += identity
        out = self.relu(out)
        
        return out
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        in_channels = 1
        self.resblock = ResBlock(in_channels,4)
        #self.resblock1 = BasicResNetBlock(2,2)
        #self.resblock2 = BasicResNetBlock(2,4)
        self.resblock3 = ResBlock(4,8)
        self.resblock4 = ResBlock(8,16)
        self.transconv = nn.ConvTranspose2d(16, 8, 3, stride=1, padding=0, output_padding=0)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.transconv2 = nn.ConvTranspose2d(8, 4, 3, stride=1, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(4)
        self.transconv3 = nn.ConvTranspose2d(4, 1, 1, stride=1, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(1)
        
    def forward(self, X):
        X = self.resblock(X)
        #X = self.resblock1(X)
        #X = self.resblock2(X)
        X = self.resblock3(X)
        X = self.resblock4(X)
        X = self.transconv(X)
        X = self.bn(X)
        X = self.relu(X)
        X = self.transconv2(X)
        X = self.bn2(X)
        X = self.relu(X)
        X = self.transconv3(X)
        X = self.bn3(X)
        X = self.relu(X)
        return X


class ResNet_Grad(nn.Module):
    def __init__(self, A):
        super(ResNet_Grad, self).__init__()
        self.A = A
        self.model = nn.Sequential()
        out_channels = 1
        conv_2_channels = None
        n_layers = 4
        for i in range(1, n_layers+1):
            self.model.append(GradBlock(self.A))
            self.model.append(ResBlock(2, 1))

    def forward(self, x):
        f_recon = self.model(x)
        return f_recon, self.head(f_recon)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.linearproj = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        else:
            self.linearproj = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.linearproj is not None:
            identity = self.linearproj(identity)
        out += identity
        out = self.relu(out)
        
        return out

class GradBlock(nn.Module):
    def __init__(self, A):
        super(GradBlock, self).__init__()
        self.A = A

    def gradient(self, f, g):
        f_cur = f
        f = f.flatten(start_dim=2).detach().permute(0, 2, 1).clone().requires_grad_(True)
        g = g.flatten(start_dim=2).detach().permute(0, 2, 1).clone()
        Af_g = (self.A @ f - g.flatten())
        ATAf_g = (self.A.T.squeeze(0) @ Af_g).permute(0, 2, 1)
        ATAf_g.mean().backward()
        out = f.grad.permute(0, 2, 1).reshape(f_cur.shape)
        return out
    def forward(self,x):
        f,g = x
        grad = self.gradient(f,g)
        return (torch.cat((f,grad),1),g)