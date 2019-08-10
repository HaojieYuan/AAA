import torch
import torch.nn as nn

class AttentionalGenerator(nn.Module):
    def __init__(self):
        super(AttentionalGenerator, self).__init__()
        # concat oringinal face with latent code.
        self.concat = Concatenator()

        self.conv0 = nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3, bias=False)
        # 2 convolution layers for down sampling.
        self.conv1 = Downsample(64, 128)
        self.conv2 = Downsample(128, 256)

        # after 2 subsampling, introduce squeeze-and-excitation operations.
        # finally, weemploy scaling to rescalse the transformation output.
        #self.ses = SqueezeExcitationScale()

        # 6 Resisual blocks, use instance normalization in generator.
        self.res1 = ResBlock(256, 256)
        self.res2 = ResBlock(256, 256)
        self.res3 = ResBlock(256, 256)
        self.res4 = ResBlock(256, 256)
        self.res5 = ResBlock(256, 256)
        self.res6 = ResBlock(256, 256)

        # 2 convolution layers for upsampling.
        self.deconv1 = Upsample(256, 128)
        self.deconv2 = Upsample(128, 64)

        self.conv3 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()


    def forward(self, x, z):
        out = self.concat(x, z)
        out = self.conv0(out)
        out = self.conv1(out)
        out = self.conv2(out)

        out = self.res1(out)
        #out = self.ses(out)
        out = self.res2(out)
        #out = self.ses(out)
        out = self.res3(out)
        #out = self.ses(out)
        out = self.res4(out)
        #out = self.ses(out)
        out = self.res5(out)
        #out = self.ses(out)
        out = self.res6(out)
        #out = self.ses(out)

        out = self.deconv1(out)
        out = self.deconv2(out)

        out = self.conv3(out)
        out = self.tanh(out)

        return out
   



class AVAE(nn.Module):
    def __init__(self):
        super(AVAE, self).__init__()
        self.res1 = ResBlock(3, 16, down_sample=True)
        self.non_local1 = NonLocal(16)
        self.res2 = ResBlock(16, 32, down_sample=True)
        self.non_local2 = NonLocal(32)
        self.res3 = ResBlock(32, 64, down_sample=True)
        self.non_local3 = NonLocal(64)
        self.res4 = ResBlock(64, 128, down_sample=True)
        self.non_local4 = NonLocal(128)
        self.fc1 = nn.Linear(128*7*7, 7)
        
    def forward(self, x):
        out = self.res1(x)
        out = self.non_local1(out)
        out = self.res2(out)
        out = self.non_local2(out)
        out = self.res3(out)
        out = self.non_local3(out)
        out = self.res4(out)
        out = self.non_local4(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        
        return out

    
class SimpleDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=112, conv_dim=64, c_dim=5, repeat_num=5):
        super(SimpleDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        #kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        #out_cls = self.conv2(h)

        #return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src
    
    
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
    def __init__(self, dim_in, dim_out):
        super(ResBlock2, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class NonLocal(nn.Module):
    def __init__(self, in_channels):
        super(NonLocal, self).__init__()
        self.downsample = Downsample(in_channels, in_channels)
        self.in1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x_reshape1 = x.view(x.shape[0], x.shape[1], -1) #[N, C, HW]
        x_reshape1 = x_reshape1.permute(0, 2, 1) #[N, HW, C]
        x_downsample = self.downsample(x) #[N, C, H/2, W/2]
        x_reshape2 = x_downsample.view(x_downsample.shape[0],
                                       x_downsample.shape[1], -1) #[N, C, HW/4]
        mul1 = torch.matmul(x_reshape1, x_reshape2) #[N, HW, HW/4]
        mul1_reshape = mul1.permute(0, 2, 1) #[N, HW/4, HW]
        mul2 = torch.matmul(x_reshape2, mul1_reshape) #[N, C, HW]
        mul2_reshape = mul2.view(*x.shape)
        
        out = self.in1(mul2_reshape)
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
