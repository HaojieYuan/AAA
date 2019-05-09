import torch
import torch.nn as nn

   
class AttentionalGenerator(nn.Module):
    def __init__(self):
        super(AttentionalGenerator, self).__init__()
        # concat oringinal face with latent code.
        self.concat = Concatenator()
        
        # 2 convolution layers for down sampling.
        self.conv1 = Downsample(10, 32)
        self.conv2 = Downsample(32, 64)
        
        # after 2 subsampling, introduce squeeze-and-excitation operations.
        # finally, weemploy scaling to rescalse the transformation output.
        self.ses = SqueezeExcitationScale()
        
        # 6 Resisual blocks, use instance normalization in generator.
        self.res1 = ResBlock(64, 128, Normalization=True)
        self.res2 = ResBlock(128, 256, Normalization=True)
        self.res3 = ResBlock(256, 512, Normalization=True)
        self.res4 = ResBlock(512, 256, Normalization=True)
        self.res5 = ResBlock(256, 128, Normalization=True)
        self.res6 = ResBlock(128, 64, Normalization=True)
        
        # 2 convolution layers for upsampling.
        self.deconv1 = Upsample(64, 32)
        self.deconv2 = Upsample(32, 3, Norm=False, activation='tanh')
        
        
    def forward(self, x, z):
        out = self.concat(x, z)
        out = self.conv1(out)
        out = self.conv2(out)
        
        out = self.res1(out)
        out = self.ses(out)
        out = self.res2(out)
        out = self.ses(out)
        out = self.res3(out)
        out = self.ses(out)
        out = self.res4(out)
        out = self.ses(out)
        out = self.res5(out)
        out = self.ses(out)
        out = self.res6(out)
        out = self.ses(out)
        
        out = self.deconv1(out)
        out = self.deconv2(out)
        
        return out


class AVAE(nn.Module):
    def __init__(self):
        super(AVAE, self).__init__()
        self.res1 = ResBlock(3, 16, down_sample=True)
        self.res2 = ResBlock(16, 32, down_sample=True)
        self.res3 = ResBlock(32, 64, down_sample=True)
        self.res4 = ResBlock(64, 128, down_sample=True)
        self.fc1 = nn.Linear(128*7*7, 7)
        
    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        
        return out

    
class SimpleDiscriminator(nn.Module):
    # PatchGAN discriminator
    pass
    
    
class SqueezeExcitationScale(nn.Module):
    def __init__(self):
        super(SqueezeExcitationScale, self).__init__()
        # same function as global avg pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sigmoid()
        
    def forward(self, x):
        weight = self.squeeze(x)
        weight = self.excitation(weight)
        out = x*weight
        
        return out
        
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 down_sample=False, Normalization=False):
        super(ResBlock, self).__init__()
        self.stride = 2 if down_sample else 1
        if down_sample:
            self.id_map = Downsample(in_channels, out_channels)
        else:
            self.id_map = conv3x3(in_channels, out_channels)
        self.Norm = Normalization
        self.conv1 = conv3x3(in_channels, out_channels, stride=self.stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        if self.Norm:
            self.in1 = nn.InstanceNorm2d(out_channels)
            self.in2 = nn.InstanceNorm2d(out_channels)
            
    def forward(self, x):
        identity = self.id_map(x)        
        out = self.conv1(x)
        if self.Norm:
            out = self.in1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        if self.Norm:
            out = self.in2(out)
        out = out + identity
        out = self.relu(out)
        
        return out
        
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=2)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, Norm=True, activation='relu'):
        super(Upsample, self).__init__()
        assert activation=='relu' or activation=='tanh'
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, 
                                          stride=2)
        self.norm = Norm
        if self.norm:
            self.in1 = nn.InstanceNorm2d(out_channels)
        if activation=='relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Tanh()
            
    def forward(self, x):
        out = self.deconv1(x)
        if self.norm:
            out = self.in1(out)
        out = self.act(out)
        
        return out[:,:,1:,1:]

class Concatenator(nn.Module):
    def __init__(self):
        super(Concatenator, self).__init__()
    
    def forward(self, x, z):
        z = z.view(*z.shape, 1, 1)
        z = z.expand(z.shape[0], z.shape[1], x.shape[-2], x.shape[-1])
        out = torch.cat((x, z), dim=1)
        
        return out

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     padding=padding, stride=stride, bias=False)
