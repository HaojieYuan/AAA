'''
F-ResNet
Reimplementation from MXNet
'''
import torch
import math
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#from torchE.nn import SyncBatchNorm2d
from .head import *
#from face_model.models import head

__all__ = ['r100_basic', 'r100_basic_affine', 'r100_basic_affine6', 'r100_basic_nofcbn', 'r100_basic_oribn', 'r100x_bottle']

BN = None

def conv3x3(in_planes, out_planes, stride=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def activation(act_type='prelu'):
    if act_type == 'prelu':
        act = nn.PReLU()
    else:
        act = nn.ReLU(inplace=True)
    return act


class BasicBlock_v3(nn.Module):
    '''
    basicblock for bn/conv/bn/act/conv/bn in ResNet manner:
    stride is assigned in conv2
    '''
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_type='prelu', use_se=False):
        super(BasicBlock_v3, self).__init__()
        bn_mom = 0.9
        m = OrderedDict()
        m['bn1'] = BN(inplanes, affine=True, eps=2e-5, momentum=bn_mom)
        m['conv1'] = conv3x3(inplanes, planes, stride=1)
        m['bn2'] = BN(planes, affine=True, eps=2e-5, momentum=bn_mom)
        m['act1'] = activation(act_type)
        m['conv2'] = conv3x3(planes, planes, stride=stride)
        m['bn3'] = BN(planes, affine=True, eps=2e-5, momentum=bn_mom)
        self.group1 = nn.Sequential(m)

        self.use_se = use_se
        if self.use_se:
            s = OrderedDict()
            s['conv1'] = nn.Conv2d(planes, planes // 16, kernel_size=1, stride=1, padding=0)
            s['act1'] = activation(act_type)
            s['conv2'] = nn.Conv2d(planes // 16, planes, kernel_size=1, stride=1, padding=0)
            s['act2'] = nn.Sigmoid()
            self.se_block = nn.Sequential(s)

        self.act2 = activation(act_type)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        if self.use_se:
            weight = F.adaptive_avg_pool2d(residual, output_size=1)
            #print(weight.size()) # (num_batch, planes, 1, 1)
            weight = self.se_block(weight)
            #print(weight.size()) # (num_batch, planes, 1, 1)
            residual = residual * weight

        out = self.group1(x) + residual
        return self.act2(out)


class Bottleneck_v3(nn.Module):
    '''
    bottleneck for bn/conv/bn/act/conv/bn/act/conv/bn in ResNet manner:
    stride is assigned in conv3
    '''
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_type='prelu', use_se=False):
        super(Bottleneck_v3, self).__init__()
        bn_mom = 0.9
        m = OrderedDict()
        m['bn1'] = BN(inplanes, affine=True, eps=2e-5, momentum=bn_mom)
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        m['bn2'] = BN(planes, affine=True, eps=2e-5, momentum=bn_mom)
        m['act1'] = activation(act_type)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        m['bn3'] = BN(planes, affine=True, eps=2e-5, momentum=bn_mom)
        m['act2'] = activation(act_type)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=stride, bias=False)
        m['bn4'] = BN(planes * 4, affine=True, eps=2e-5, momentum=bn_mom)
        self.group1 = nn.Sequential(m)

        self.use_se = use_se
        if self.use_se:
            s = OrderedDict()
            s['conv1'] = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1, padding=0)
            s['act1'] = activation(act_type)
            s['conv2'] = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1, padding=0)
            s['act2'] = nn.Sigmoid()
            self.se_block = nn.Sequential(s)

        self.act2 = activation(act_type)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        if self.use_se:
            weight = F.adaptive_avg_pool2d(residual, output_size=1)
            #print(weight.size()) # (num_batch, planes, 1, 1)
            weight = self.se_block(weight)
            #print(weight.size()) # (num_batch, planes, 1, 1)
            residual = residual * weight

        out = self.group1(x) + residual
        return self.act2(out)


class Bottleneck_v3_x(nn.Module):
    '''
    bottleneck for bn/conv/bn/act/conv/bn/act/conv/bn in ResNXet manner:
    stride is assigned in conv3
    '''
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, act_type='prelu', use_se=False):
        super(Bottleneck_v3_x, self).__init__()
        bn_mom = 0.9
        num_group = 32
        m = OrderedDict()
        m['bn1'] = BN(inplanes, affine=True, eps=2e-5, momentum=bn_mom)
        m['conv1'] = nn.Conv2d(inplanes, planes, groups=num_group, kernel_size=1,
                               stride=1, bias=False)
        m['bn2'] = BN(planes, affine=True, eps=2e-5, momentum=bn_mom)
        m['act1'] = activation(act_type)
        m['conv2'] = nn.Conv2d(planes, planes, groups=num_group, kernel_size=3,
                               stride=1, padding=1, bias=False)
        m['bn3'] = BN(planes, affine=True, eps=2e-5, momentum=bn_mom)
        m['act2'] = activation(act_type)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=stride, bias=False)
        m['bn4'] = BN(planes * 4, affine=True, eps=2e-5, momentum=bn_mom)
        self.group1 = nn.Sequential(m)

        self.use_se = use_se
        if self.use_se:
            s = OrderedDict()
            s['conv1'] = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1, padding=0)
            s['act1'] = activation(act_type)
            s['conv2'] = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1, padding=0)
            s['act2'] = nn.Sigmoid()
            self.se_block = nn.Sequential(s)

        self.act2 = activation(act_type)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        if self.use_se:
            weight = F.adaptive_avg_pool2d(residual, output_size=1)
            #print(weight.size()) # (num_batch, planes, 1, 1)
            weight = self.se_block(weight)
            #print(weight.size()) # (num_batch, planes, 1, 1)
            residual = residual * weight

        out = self.group1(x) + residual
        return self.act2(out)

class ResNet(nn.Module):
    '''
    Designed for 112x112 input, keep higher feature map:
    the first conv is set to be (kernel=3, stride=1),
    remove maxpooling in the first stage.
    fewer channels [64, 128, 256, 512].
    '''
    def __init__(self, block, layers, feature_dim, act_type, fc_type='E',
                 group_size=1, group=None, sync_stats=False, use_sync_bn=False):

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.bn_mom = 0.9

        global BN
        def BNFunc(*args, **kwargs):
            return SyncBatchNorm2d(*args, **kwargs, group_size=group_size,
                                   group=group, sync_stats=sync_stats)
        if use_sync_bn:
            BN = nn.BatchNorm2d
        else:
            BN = nn.BatchNorm2d


        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m['bn1'] = BN(64, affine=True, eps=2e-5, momentum=self.bn_mom)
        m['act1'] = activation(act_type)
        self.group1 = nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if fc_type == 'E':
            self.fc1 = get_fc_E(BN, 512, 7, 7, feature_dim)
        elif fc_type == 'F':
            self.fc1 = get_fc_F(BN, 512, 7, 7, feature_dim)
        elif fc_type == 'G':
            self.fc1 = get_fc_G(BN, 512, 7, 7, feature_dim)
        elif fc_type == 'H':
            self.fc1 = get_fc_H(BN, 512, 7, 7, feature_dim)
        else:
            raise RuntimeError('fc_type not supported!')
        #self.fc1 = head.__dict__['get_fc_'+fc_type](BN, 512, 7, 7, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_normal(m.weight.data)
                fan_in = m.out_channels * m.kernel_size[0] * m.kernel_size[1]
                scale = math.sqrt(2. / fan_in)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.zeros_()
            # Xavier can not be applied to less than 2D.
            elif isinstance(m, nn.BatchNorm2d):
                if not m.weight is None:
                    m.weight.data.fill_(1)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                scale = math.sqrt(3. / m.in_features)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    m.bias.data.uniform_(-scale, scale)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion, affine=True,
                               eps=2e-5, momentum=self.bn_mom),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)
        #print(x.size())
        x = self.layer1(x)
        #print(x.size())
        x = self.layer2(x)
        #print(x.size())
        x = self.layer3(x)
        #print(x.size())
        x = self.layer4(x)
        #print(x.size())
        x = self.fc1(x)
        #print(x.size())
        return x


def r100_basic(feature_dim, group_size=1, group=None, sync_stats=False):

    model = ResNet(BasicBlock_v3, [3, 13, 30, 3], feature_dim=feature_dim,
                   act_type='prelu', fc_type='E', group_size=group_size,
                   group=group, sync_stats=sync_stats, use_sync_bn=True)
    return model

def r100_basic_affine(feature_dim, group_size=1, group=None, sync_stats=False):

    model = ResNet(BasicBlock_v3, [3, 13, 30, 3], feature_dim=feature_dim,
                   act_type='prelu', fc_type='G', group_size=group_size,
                   group=group, sync_stats=sync_stats, use_sync_bn=True)
    return model

def r100_basic_affine6(feature_dim, group_size=1, group=None, sync_stats=False):

    model = ResNet(BasicBlock_v3, [3, 13, 30, 3], feature_dim=feature_dim,
                   act_type='prelu', fc_type='H', group_size=group_size,
                   group=group, sync_stats=sync_stats, use_sync_bn=True)
    return model


def r100_basic_nofcbn(feature_dim, group_size=1, group=None, sync_stats=False):

    model = ResNet(BasicBlock_v3, [3, 13, 30, 3], feature_dim=feature_dim,
                   act_type='prelu', fc_type='F', group_size=group_size,
                   group=group, sync_stats=sync_stats, use_sync_bn=True)
    return model


def r100_basic_oribn(feature_dim, group_size=1, group=None, sync_stats=False):

    model = ResNet(BasicBlock_v3, [3, 13, 30, 3], feature_dim=feature_dim,
                   act_type='prelu', fc_type='E', use_sync_bn=False)
    return model


# def r100_bottle(feature_dim, group_size=1, group=None, sync_stats=False, pretrain=False):
#     model = ResNet(Bottleneck_v3, [3, 13, 30, 3], num_classes=feature_dim,
#                    act_type='prelu', fc_type='E')
#     return model


def r100x_bottle(feature_dim, group_size=1, group=None, sync_stats=False):

    model = ResNet(Bottleneck_v3_x, [3, 13, 30, 3], feature_dim=feature_dim,
                   act_type='prelu', fc_type='E', group_size=group_size,
                   group=group, sync_stats=sync_stats)
    return model


if __name__ == '__main__':
    inputs = torch.autograd.Variable(torch.randn(4, 3, 112, 112), volatile=True)

    model = r100_basic(feature_dim=512)
    #model = r100x_bottle(num_classes=100, emb_size=512)
    #print(model)
    outputs = model(inputs)
    #print(outputs.size())
