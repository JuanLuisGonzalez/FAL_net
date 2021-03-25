import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Define VGG19
class Vgg19_pc(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_pc, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        vgg_pretrained_features = nn.DataParallel(vgg_pretrained_features.cuda())

        # This has Vgg config E:
        # partial convolution paper uses up to pool3
        # [64,'r', 64,r, 'M', 128,'r', 128,r, 'M', 256,'r', 256,r, 256,r, 256,r, 'M', 512,'r', 512,r, 512,r, 512,r]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        n_new = 0
        for x in range(5):  # pool1,
            self.slice1.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(5, 10):  # pool2
            self.slice2.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(10, 19):  # pool3
            self.slice3.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for x in range(19, 28):  # pool4
            self.slice4.add_module(str(n_new), vgg_pretrained_features.module[x])
            n_new += 1
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, full=False):
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_4 = self.slice3(h_relu2_2)
        if full:
            h_relu4_4 = self.slice4(h_relu3_4)
            return h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4
        else:
            return h_relu1_2, h_relu2_2, h_relu3_4


# Get an instance of the pre-trained VGG19
vgg = Vgg19_pc()


# Our loss functions
def rec_loss_fnc(mask, synth, label, vgg_label, a_p):
    loss = torch.mean(mask * torch.abs(synth - label))
    if a_p > 0 and vgg_label is not None:
        loss = loss + a_p * perceptual_loss(vgg(mask * synth + (1 - mask) * label), vgg_label)
    return loss


def perceptual_loss(out_vgg, label_vgg, layer=None):
    if layer is not None:
        l_p = torch.mean((out_vgg[layer] - label_vgg[layer]) ** 2)
    else:
        l_p = 0
        for i in range(3):
            l_p += torch.mean((out_vgg[i] - label_vgg[i]) ** 2)

    return l_p


def smoothness(img, disp, gamma=1):
    B, C, H, W = img.shape

    m_rgb = torch.ones((B, C, 1, 1)).cuda()
    m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
    m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
    m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
    gray_img = getGrayscale(img + m_rgb)

    # Disparity smoothness
    sx_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    sy_filter = torch.autograd.Variable(torch.Tensor([[0, -1, 0], [0, 2, 0], [0, -1, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dy_filter = torch.autograd.Variable(torch.Tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx1_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dy1_filter = torch.autograd.Variable(torch.Tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])).unsqueeze(
        0).unsqueeze(0).cuda()
    dx_img = F.conv2d(gray_img, sx_filter, padding=1, stride=1)
    dy_img = F.conv2d(gray_img, sy_filter, padding=1, stride=1)
    dx_d = F.conv2d(disp, dx_filter, padding=1, stride=1)
    dy_d = F.conv2d(disp, dy_filter, padding=1, stride=1)
    dx1_d = F.conv2d(disp, dx1_filter, padding=1, stride=1)
    dy1_d = F.conv2d(disp, dy1_filter, padding=1, stride=1)
    Cds = torch.mean(
        (torch.abs(dx_d) + torch.abs(dx1_d)) * torch.exp(-gamma * torch.abs(dx_img)) +
        (torch.abs(dy_d) + torch.abs(dy1_d)) * torch.exp(-gamma * torch.abs(dy_img)))
    return Cds


def getGrayscale(input):
    # Input is mini-batch N x 3 x H x W of an RGB image (analog rgb from 0...1)
    output = torch.autograd.Variable(input.data.new(*input.size()))
    # Output is mini-batch N x 3 x H x W from y = 0 ... 1
    output[:, 0, :, :] = 0.299 * input[:, 0, :, :] + 0.587 * input[:, 1, :, :] + 0.114 * input[:, 2, :, :]
    return output[:, 0, :, :].unsqueeze(1)


###################################################################################################################
# From https://github.com/ClementPinard/FlowNetPytorch
###################################################################################################################
class MultiscaleEPE(nn.Module):
    def __init__(self, multiscale_weights):
        super(MultiscaleEPE, self).__init__()
        self.w_m = multiscale_weights

    def forward(self, output_diparity, label_disparity):
        return multiscaleEPE(output_diparity, label_disparity, self.w_m, False)


def EPE(net_out, target, sparse=False, disp=True, mean=True):
    EPE_map = torch.norm(target - net_out, p=2, dim=1)  # l2 norm per image
    batch_size = EPE_map.size(0)
    if sparse:
        if disp:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = target[:, 0] == 0
        else:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = (target[:, 0] == 0) & (target[:, 1] == 0)
        EPE_map = EPE_map[~mask.data]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size


def sparse_max_pool(input, size):
    positive = (input > 0).float()
    negative = (input < 0).float()
    output = nn.functional.adaptive_max_pool2d(input * positive, size) - nn.functional.adaptive_max_pool2d(
        -input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):
        b, _, h, w = output.size()
        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = nn.functional.adaptive_avg_pool2d(target, (h, w))
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]  # if output is not a touple, make it a touple of one element
    if weights is None:
        weights = [0.001, 0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert (len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)  # linear combination of net outputs and weights
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
    return EPE(upsampled_output, target, sparse, mean=True)
