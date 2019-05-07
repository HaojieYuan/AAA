import torch
import torch.nn as nn

def conv1x1(in_channels, out_channels):
	# 1x1 convolution without downsample.
	# no bias here because of Instance Normalization.
	return nn.Conv2d(in_channels, out_channels, kernel_size=1,
		             padding=0, stride=1, bias=False)

class NonLocalBlock(nn.module):
	""" CihangXie Version of NonLocalBlock """
	def __init__(self, in_channels):
		super(NonLocalBlock, self).__init__()
		self.conv1 = conv1x1(in_channels, in_channels)
		self.in1 = nn.InstanceNorm2d(in_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		# B for batch size, C for channel, H for Height, W for Width
		identity = x #[B, C, H, W]
		x_ch_first = x.view([x.shape[0], x.shape[1], -1]) #[B, C, HW]
		x_ch_last = x_ch_first.permute(0, 2, 1) #[B, HW, C]
		
		mul1 = torch.matmul(x_ch_last, x_ch_first) #[B, HW, HW]
		mul2 = torch.matmul(x_ch_first, mul1) #[B, C, HW]
		mul_out = mul2.view(x.shape) #[B, C, H, W]
		
		out = self.conv1(mul_out)
		out = self.in1(out)
		out = out + indentity
		out = self.relu(out)

		return out
