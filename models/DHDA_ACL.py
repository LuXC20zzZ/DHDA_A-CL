import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class AdversarialNet(nn.Module):
    def __init__(self):
        super(AdversarialNet, self).__init__()
        self.Adv_Net = nn.Sequential(
            nn.Linear(960, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(),

            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(),

            nn.Linear(256, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.Adv_Net(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2))  # 1.Matmul
        u = u / self.scale  # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)  # 3.Mask

        attn = self.softmax(u)  # 4.Softmax
        output = torch.bmm(attn, v)  # 5.Output

        return attn, output


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)
        output = self.fc_o(output)

        return attn, output


class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return attn, output


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, stride=2)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)
        out = self.pool(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel=1, out_channel=10, zero_init_residual=False):
        super(ResNet, self).__init__()

        self.inplanes = 16

        self.WD_conv = nn.Sequential(
            nn.Conv1d(in_channel, self.inplanes, kernel_size=64, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 16, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(16)
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, stride=2)

        self.layer1 = self._make_layer(block, 16, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 32, 64, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 64, 64, layers[3], stride=1)

        self.mhsa = SelfAttention(n_head=1, d_k=128, d_v=64, d_x=15, d_o=15)

        self.mlp = nn.Sequential(
            nn.Linear(960, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5)
        )

        self.Adv_layer = AdversarialNet()

        self.fc1 = nn.Sequential(
            nn.Linear(960, 100),
            # nn.BatchNorm1d(100),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(100, 100),
            # nn.BatchNorm1d(100),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.fc2_1 = nn.Sequential(
            nn.Linear(960, 100),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
        )

        self.fc2_2 = nn.Sequential(
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
        )

        self.fc2_3 = nn.Sequential(
            nn.Linear(100, out_channel),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, alpha):
        x_WD = self.WD_conv(x)
        x_WD = self.relu(x_WD)
        x_WD = self.pool(x_WD)

        x1 = self.layer1(x_WD)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        mask = torch.zeros(x4.size(0), 64, 64).bool().cuda()
        _, x_mhsa = self.mhsa(x4, mask=mask)

        out = x4.view(x4.size(0), -1)
        out_att = x_mhsa.view(x_mhsa.size(0), -1)

        out1 = self.fc1(out)

        out2_1 = self.fc2_1(out)
        out2_2 = self.fc2_2(out2_1)
        out2_3 = self.fc2_3(out2_2)

        out_ = self.mlp(out)
        out_att_ = self.mlp(out_att)

        in_adv = out * out2_3
        in_adv = ReverseLayerF.apply(in_adv, alpha)
        out_adv = self.Adv_layer(in_adv)

        return out, out_, out_att_, out1, out_adv, out2_3


def Model(**kwargs):
    """
    Constructs a modified ResNet model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model