# FAL_netC.py This script contains the network architecture of FAL_netC
# Copyright (C) 2021  Juan Luis Gonzalez Bello (juanluisgb@kaist.ac.kr)
# This software is not for commercial use
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["FAL_netC"]


def FAL_netC(data=None, no_levels=33):
    model = FAL_net(batchNorm=False, no_levels=no_levels)
    if data is not None:
        model.load_state_dict(data["state_dict"])
    return model


def conv_elu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, pad=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ELU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                bias=True,
            ),
            nn.ELU(inplace=True),
        )


class deconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(deconv, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x, ref):
        x = F.interpolate(x, size=(ref.size(2), ref.size(3)), mode="nearest")
        x = self.elu(self.conv1(x))
        return x


def predict_rgb(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 3, kernel_size=3, stride=1, padding=1, bias=False),
    )


class residual_block(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(residual_block, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x):
        x = self.elu(self.conv2(self.elu(self.conv1(x))) + x)
        return x


def predict_amask(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, in_planes // 2, kernel_size=3, stride=1, padding=1, bias=True
        ),
        nn.ELU(inplace=True),
        nn.Conv2d(
            in_planes // 2, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        ),
        nn.Sigmoid(),
    )


class BackBone(nn.Module):
    def __init__(self, batchNorm=True, no_in=3, no_flow=1, no_out=64):
        super(BackBone, self).__init__()
        self.batchNorm = batchNorm

        # Encoder
        self.batchNorm = batchNorm
        self.conv0 = conv_elu(self.batchNorm, no_in, 32, kernel_size=3)
        self.conv0_1 = residual_block(32)
        self.conv1 = conv_elu(self.batchNorm, 32 + no_flow, 64, pad=1, stride=2)
        self.conv1_1 = residual_block(64)
        self.conv2 = conv_elu(self.batchNorm, 64, 128, pad=1, stride=2)
        self.conv2_1 = residual_block(128)
        self.conv3 = conv_elu(self.batchNorm, 128, 256, pad=1, stride=2)
        self.conv3_1 = residual_block(256)
        self.conv4 = conv_elu(self.batchNorm, 256, 256, pad=1, stride=2)
        self.conv4_1 = residual_block(256)
        self.conv5 = conv_elu(self.batchNorm, 256, 512, pad=1, stride=2)
        self.conv5_1 = residual_block(512)
        self.conv6 = conv_elu(self.batchNorm, 512, 512, pad=1, stride=2)
        self.conv6_1 = residual_block(512)
        self.elu = nn.ELU(inplace=True)

        # i and up convolutions
        self.deconv6 = deconv(512, 256)
        self.iconv6 = conv_elu(self.batchNorm, 256 + 512, 512)
        self.deconv5 = deconv(512, 256)
        self.iconv5 = conv_elu(self.batchNorm, 256 + 256, 256)
        self.deconv4 = deconv(256, 128)
        self.iconv4 = conv_elu(self.batchNorm, 128 + 256, 256)
        self.deconv3 = deconv(256, 128)
        self.iconv3 = conv_elu(self.batchNorm, 128 + 128, 128)
        self.deconv2 = deconv(128, 64)
        self.iconv2 = conv_elu(self.batchNorm, 64 + 64, 64)
        self.deconv1 = deconv(64, 64)
        self.iconv1 = nn.Conv2d(
            32 + 64, no_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.amask_conv = predict_amask(32 + 64, 1)

        # Initialize conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight.data
                )  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, flow, feats=None):
        _, _, H, W = x.shape

        # Encoder section
        out_conv0 = self.conv0_1(self.conv0(x))
        out_conv1 = self.conv1_1(self.conv1(torch.cat((out_conv0, flow), 1)))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_deconv6 = self.deconv6(out_conv6, out_conv5)
        concat6 = torch.cat((out_deconv6, out_conv5), 1)
        iconv6 = self.iconv6(concat6)

        out_deconv5 = self.deconv5(iconv6, out_conv4)
        concat5 = torch.cat((out_deconv5, out_conv4), 1)
        iconv5 = self.iconv5(concat5)

        out_deconv4 = self.deconv4(iconv5, out_conv3)
        concat4 = torch.cat((out_deconv4, out_conv3), 1)
        iconv4 = self.iconv4(concat4)

        out_deconv3 = self.deconv3(iconv4, out_conv2)
        concat3 = torch.cat((out_deconv3, out_conv2), 1)
        iconv3 = self.iconv3(concat3)

        out_deconv2 = self.deconv2(iconv3, out_conv1)
        concat2 = torch.cat((out_deconv2, out_conv1), 1)
        iconv2 = self.iconv2(concat2)

        out_deconv1 = self.deconv1(iconv2, out_conv0)
        concat1 = torch.cat((out_deconv1, out_conv0), 1)
        dlog = self.iconv1(concat1)

        return dlog


class FAL_net(nn.Module):
    def __init__(self, batchNorm, no_levels):
        super(FAL_net, self).__init__()
        self.no_levels = no_levels
        self.no_fac = 1
        self.synth = BackBone(batchNorm, no_in=3, no_flow=1, no_out=self.no_levels)
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # An additional 1x1 conv layer on the logits (not shown in paper). Its contribution should not be much.
        self.conv0 = nn.Conv2d(
            self.no_levels,
            self.no_fac * self.no_levels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        nn.init.kaiming_normal_(
            self.conv0.weight.data
        )  # initialize weigths with normal distribution
        self.conv0.bias.data.zero_()  # initialize bias as zero

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]

    def forward(
        self,
        input_left,
        min_disp,
        max_disp,
        ret_disp=True,
        ret_subocc=False,
        ret_pan=False,
    ):
        B, C, H, W = input_left.shape

        # convert into relative value
        x_pix_min = 2 * min_disp / W
        x_pix_max = 2 * max_disp / W

        # convert into 1 channel feature
        flow = torch.ones(B, 1, H, W).type(input_left.type())
        flow[:, 0, :, :] = max_disp * flow[:, 0, :, :] / 100  # normalized?

        # Synthesize zoomed image
        dlog = self.synth(input_left, flow)

        # Get shifted dprob
        dlog0 = self.conv0(dlog)
        sm_dlog0 = self.softmax(dlog0)

        # Get disp and conf mask from kernel
        if ret_disp:
            disp = 0

            for n in range(0, self.no_levels * self.no_fac):
                with torch.no_grad():
                    c = n / (self.no_levels * self.no_fac - 1)  # Goes from 0 to 1
                    w = max_disp * torch.exp(torch.log(max_disp / min_disp) * (c - 1))
                disp = disp + w.unsqueeze(1) * sm_dlog0[:, n, :, :].unsqueeze(1)

        if ret_disp and not ret_subocc and not ret_pan:
            return disp

        i_tetha = torch.zeros(B, 2, 3).cuda()
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        i_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=True)

        for n in range(0, self.no_levels * self.no_fac):
            with torch.no_grad():
                # Exponential quantization
                c = n / (self.no_levels * self.no_fac - 1)  # Goes from 0 to 1
                # x_of = c * (x_pix_max - x_pix_min) + x_pix_min # This is linear quantization
                x_of = x_pix_max * torch.exp(torch.log(x_pix_max / x_pix_min) * (c - 1))
                out_grid = i_grid.clone()
                out_grid[:, :, :, 0] = out_grid[:, :, :, 0] + x_of
            if n == 0:
                Dprob = F.grid_sample(
                    dlog0[:, n, :, :].unsqueeze(1), out_grid, align_corners=True
                )
            else:
                Dprob = torch.cat(
                    (
                        Dprob,
                        F.grid_sample(
                            dlog0[:, n, :, :].unsqueeze(1), out_grid, align_corners=True
                        ),
                    ),
                    1,
                )
        Dprob = self.softmax(Dprob)

        # Blend shifted features
        p_im0 = 0
        maskR = 0
        maskL = 0
        for n in range(0, self.no_levels * self.no_fac):
            with torch.no_grad():
                # Exponential quantization
                c = n / (self.no_levels * self.no_fac - 1)  # Goes from 0 to 1
                # x_of = c * (x_pix_max - x_pix_min) + x_pix_min # This is linear quantization
                x_of = x_pix_max * torch.exp(torch.log(x_pix_max / x_pix_min) * (c - 1))
                out_grid = i_grid.clone()
                out_grid[:, :, :, 0] = out_grid[:, :, :, 0] + x_of

                if ret_subocc:
                    # Get content visible in right that is also visible in left
                    maskR = maskR + F.grid_sample(
                        sm_dlog0[:, n, :, :].unsqueeze(1).detach(),
                        out_grid,
                        align_corners=True,
                    )

                    # Get content visible in left that is also visible in right
                    out_grid1 = i_grid.clone()
                    out_grid1[:, :, :, 0] = out_grid1[:, :, :, 0] - x_of
                    maskL = maskL + F.grid_sample(
                        Dprob[:, n, :, :].unsqueeze(1).detach(),
                        out_grid1,
                        align_corners=True,
                    )

            # Get synth right view
            if ret_pan:
                p_im0 = p_im0 + F.grid_sample(
                    input_left, out_grid, align_corners=True
                ) * Dprob[:, n, :, :].unsqueeze(1)

        # Return selected outputs (according to input arguments)
        output = []
        if ret_pan:
            output.append(p_im0)
        if ret_disp:
            output.append(disp)
        if ret_subocc:
            maskR[maskR > 1] = 1
            maskL[maskL > 1] = 1
            output.append(maskL)
            output.append(maskR)

        return output
