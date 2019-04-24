import torch
import torch.nn as nn

__all__ = ['get_fc_A', 'get_fc_E', 'get_fc_F', 'get_fc_G', 'get_fc_H']

class get_fc_A(nn.Module):
    def __init__(self, BN, in_feature, in_h, in_w, out_feature):
        super(get_fc_A, self).__init__()
        self.dropout = nn.Dropout2d(p=0.5, inplace=True)
        self.fc = nn.Linear(in_feature*in_h*in_w, out_feature)
    def forward(self, x):
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class get_fc_E(nn.Module):
    def __init__(self, BN, in_feature, in_h, in_w, out_feature):
        super(get_fc_E, self).__init__()
        self.bn1 = BN(in_feature, affine=True, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(in_feature*in_h*in_w, out_feature)
        #TODO here momentum should be 0.1!
        self.bn2 = nn.BatchNorm1d(out_feature, affine=False, eps=2e-5, momentum=0.9)
    def forward(self, x):
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        return x

class get_fc_F(nn.Module):
    def __init__(self, BN, in_feature, in_h, in_w, out_feature):
        super(get_fc_F, self).__init__()
        self.bn1 = BN(in_feature, affine=True, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(in_feature*in_h*in_w, out_feature)
    def forward(self, x):
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class get_fc_G(nn.Module):
    def __init__(self, BN, in_feature, in_h, in_w, out_feature):
        super(get_fc_G, self).__init__()
        self.bn1 = BN(in_feature, affine=True, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(in_feature*in_h*in_w, out_feature)
        #TODO here momentum should be 0.1!
        self.bn2 = nn.BatchNorm1d(out_feature, affine=True, eps=2e-5, momentum=0.9)
        self.bn2.weight.data.fill_(4)

    def forward(self, x):
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        return x

class get_fc_H(nn.Module):
    def __init__(self, BN, in_feature, in_h, in_w, out_feature):
        super(get_fc_H, self).__init__()
        self.bn1 = BN(in_feature, affine=True, eps=2e-5, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(in_feature*in_h*in_w, out_feature)
        #TODO here momentum should be 0.1!
        self.bn2 = nn.BatchNorm1d(out_feature, affine=True, eps=2e-5, momentum=0.9)
        self.bn2.weight.data.fill_(6)

    def forward(self, x):
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        return x

