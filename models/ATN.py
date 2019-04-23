import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels):
    # Do not need bias due to Instance normalization.
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     padding=1, stride=1, bias=False)

class ResBlock(nn.Module):
    """ Residual block with same input & output size """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels, out_channels)
        self.in2 = nn.InstanceNorm2d(out_channels)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(conv3x3(in_channels, out_channels),
                                            nn.InstanceNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out = out + indentity
        out = self.relu(out)
        return out

class FeatureExtract(nn.Module):
    """ Feature extract network for source & Target img"""
    def __init__(self, out_channels):
        # out_channels here shoudl be a list
        # it is resblock channel config
        super(FeatureExtract, self).__init__()
        in_layer = nn.InstanceNorm2d
        self.conv1 = conv3x3(3, out_channels[0])
        self.in1 = in_layer(out_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = [ResBlock(out_channels[i], out_channels[i+1]) \
                                    for i in range(len(out_channels)-1) ]

    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        for res_block in self.res_blocks:
            out = res_block(out)
        return out

class ATN(nn.Module):
    """ Adversarial Transform Network"""
    def __init__(self, down_sample=False, parallel=True, input_size=112):
        super(ATN, self).__init__()
        self.fe1 = FeatureExtract([64, 128, 256, 256])
        self.fe2 = FeatureExtract([64, 128, 256, 256])
        def concat_channel(tensor_list):
            return torch.cat(tensor_list, dim=1)
        self.concat = concat_channel
        self.res1 = ResBlock(256, 512)
        self.res2 = ResBlock(512, 512)
        self.res3 = ResBlock(512, 256)
        self.res4 = ResBlock(256, 128)
        self.res5 = ResBlock(128, 64)
        self.conv1 = conv3x3(64, 3)
        self.tanh = nn.Tanh()

    def forward(self, source, target):
        source_feature = self.fe1(source)
        target_feature = self.fe2(target)
        combined_feature = self.concat((source_feature, target_feature))
        out = self.res1(combined_feature)
        out = self.res2(combined_feature)
        out = self.res3(combined_feature)
        out = self.res4(combined_feature)
        out = self.res5(combined_feature)
        out = self.conv1(out)
        out = self.tanh(out)

        return out

