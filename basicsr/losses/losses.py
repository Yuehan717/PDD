from cProfile import label
from ctypes import resize
import math
from mimetypes import init
from multiprocessing import reduction
from turtle import forward
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.archs.alex_arch import AlexFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils import get_root_logger
from basicsr.archs.arch_util import DWT_Haar_split
from .loss_util import weighted_loss
from einops import rearrange
from torch.autograd import Variable
from math import exp

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


class CrossEntropyLoss(nn.Module):
    """cross-entropy loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', one_hot=False, S2B=False, eps=1e-6):
        super(CrossEntropyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.one_hot = one_hot
        self.S2B = S2B
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        if self.S2B:
            pred = rearrange(pred, 'b c h w -> (b h w) c')
            target = rearrange(target, 'b c h w -> (b h w) c')
        else:
            pred = pred.flatten(1)
            target = target.flatten(1)
        if self.one_hot:
            ce_loss = F.cross_entropy(pred, target,reduction=self.reduction)
        else:
            pred=torch.softmax(pred, dim=-1)
            target=torch.softmax(target, dim=-1)
            ce_loss = torch.mean(-torch.sum(target * torch.log(pred+self.eps), 1))
        return self.loss_weight * ce_loss
    
@LOSS_REGISTRY.register()
class KL(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, kl_loss_weight=1.0, reduction='mean'):
        super(KL, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        # self.pix_loss_weight = pix_loss_weight
        self.kl_loss_weight = kl_loss_weight
        # self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        _, dist = pred
        # pix_loss = self.pix_loss_weight * l1_loss(pred_rec, target, weight, reduction=self.reduction)
        mu, log_var = dist
        kl_loss = self.kl_loss_weight * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0).mean()
                
        return kl_loss


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CosineSimilarity(nn.Module):
    """
    Squared Exponential Covariance Function
    """
    def __init__(self, l=0.5, sigmaf=0.5, S2B=False):
        super(CosineSimilarity, self).__init__()
        self.func = nn.CosineSimilarity(dim=1)
        self.S2B = S2B
    def forward(self,x1,x2):
        # shape of input: B, C, H, W
        B, C, _ ,_ = x1.size()
        if self.S2B:
            x1 = rearrange(x1, 'b c h w -> (b h w) c')
            x2 = rearrange(x2, 'b c h w -> (b h w) c')
        else:
            x1 = x1.flatten(1)
            x2 = x2.flatten(1)
        similarity = self.func(x1,x2).mean()

        return similarity

@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)

@LOSS_REGISTRY.register()
class WeightedWvtLoss(L1Loss):
    def __init__(self, loss_weight=1, reduction='mean', weights=[1, 0.5, 0.1]):
        super(WeightedWvtLoss, self).__init__(loss_weight, reduction)
        self.dwt = DWT_Haar_split()
        self.weights = weights
    def forward(self, x1, x2):
        L1, M1, H1 = self.dwt(x1)
        L2, M2, H2 = self.dwt(x2)
        loss_L = super().forward(L1, L2)
        loss_M = super().forward(M1, M2)
        loss_H = super().forward(H1, H2)
        return self.weights[0] * loss_L + self.weights[1] * loss_M + self.weights[2] * loss_H

@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # if isinstance(x, tuple):
        #     x, _, _ = x
        # c,h,w = x.shape[2:]
        # x = x.view(-1,c,h,w)
        # gt = gt.view(-1,c,h,w)
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        # if isinstance(input, tuple):
        #     input, _, _ = input
        # c,h,w = input.shape[2:]
        # input = input.view(-1,c,h,w)
        # gt = gt.view(-1,c,h,w)
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight




def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean'):
        super(GANFeatLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'charbonnier':
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|charbonnier')

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_d = len(pred_fake)
        loss = 0
        for i in range(num_d):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_d
        return loss * self.loss_weight


def F_norm(m):
    # B*T, c, h, w = m.size()
    # F_m = torch.sqrt(torch.square(m).sum(dim=(1,2,3),keepdim=True)) # size of B
    norm = torch.linalg.norm(m, ord=2, dim=1, keepdim=True)
    # print(f"normalization value {norm}")
    m = m/(norm + 1e-8)
    return m

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    shape = x.shape
    if len(shape) == 2 :
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    elif len(shape) == 3:
        L, n, m = shape
        assert n==m
        return x.flatten(start_dim=1)[:,:-1].view(L, n-1, n+1)[:,:,1:].flatten(start_dim=1) 


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
@LOSS_REGISTRY.register()
class SSIMLoss(torch.nn.Module):
    def __init__(self, loss_weight = 1.0 ,window_size = 11, size_average = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.loss_weight = loss_weight
    def forward(self, img1, img2):
        if len(img1.size())==5:
            img1 = rearrange(img1, 'b n c h w -> (b n) c h w')
            img2 = rearrange(img2, 'b n c h w -> (b n) c h w')
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel


        return -1 * _ssim(img1, img2, window, self.window_size, channel, self.size_average) * self.loss_weight

        
@LOSS_REGISTRY.register()
class DistillationLoss(nn.Module):
    """Distillation loss with VGG feature extractor.

    Args:
        layer_weights_{inter/intra} (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        model_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        {rinter/rinter}_weight (float): the weight for each regularization term.
            Default: 1.0
        delta_inter_type (str): function for calculating inter-model distance
            Default: 'gram_change'
        delta_intra_type (str): function for calculating intra-model distance
            Default: 'l1'
        cri_inter (str): function for difference between inter-model distances
            Default: 'fro'
        cri_intra (str): function for difference between intra-model distances
            Default: 'ce' 
    """

    def __init__(self,
                 layer_weights_inter,
                 layer_weights_intra,
                 model_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 rinter_weight=1.0,
                 rintra_weight=1.0,
                 delta_inter_type='gram_change',
                 delta_intra_type='l1',
                 cri_inter='fro',
                 cri_intra='l1'):
        super(DistillationLoss, self).__init__()
        self.rinter_weight = rinter_weight
        self.rintra_weight = rintra_weight
        self.layer_weights_inter = layer_weights_inter
        self.layer_weights_intra = layer_weights_intra
        self.layer_weights = dict(layer_weights_inter, **layer_weights_intra)
        if 'vgg' in model_type:
            self.model = VGGFeatureExtractor(
                layer_name_list=list(self.layer_weights.keys()),
                vgg_type=model_type,
                use_input_norm=use_input_norm,
                range_norm=range_norm)
        else:
            self.model = AlexFeatureExtractor(
                layer_name_list=list(self.layer_weights.keys()),
                use_input_norm=use_input_norm,
                range_norm=range_norm)

        self.delta_inter_type = delta_inter_type
        self.delta_intra_type = delta_intra_type
        self.cri_inter = cri_inter
        self.cri_intra = cri_intra
        
        #### define appearance loss
        if self.delta_inter_type == 'l1':
            self.delta_inter = torch.nn.L1Loss(reduction='none')
        elif self.delta_inter_type == 'l2':
            self.delta_inter = torch.nn.MSELoss(reduction='none')
        elif 'gram' in self.delta_inter_type:
            self.delta_inter = None
        else:
            raise NotImplementedError(f'{delta_inter_type} criterion has not been supported.')
        
        if self.cri_inter == 'fro':
            self.criterion_inter = None
        else:
            raise NotImplementedError(f'{cri_inter} criterion has not been supported.')
        
        #### define content loss ###
        if self.delta_intra_type == 'l1':
            self.delta_intra = torch.nn.L1Loss(reduction='none')
        elif self.delta_intra_type == 'fro':
            self.delta_intra = None
        else:
            raise NotImplementedError(f'{delta_intra_type} criterion has not been supported.')
        
        if self.cri_intra == 'l1':
            self.criterion_intra = torch.nn.L1Loss(reduction='mean')
        elif self.cri_intra == 'l2':
            self.criterion_intra = torch.nn.MSELoss(reduction='mean')
        elif self.cri_intra == 'fro':
            self.criterion_intra = None
        elif self.cri_intra == 'ce':
            self.criterion_intra = CrossEntropyLoss(one_hot=False, S2B=True)
        elif self.cri_intra == 'cosine_sim':
            self.criterion_intra = CosineSimilarity(S2B=True)
        else:
            raise NotImplementedError(f'{cri_intra} criterion has not been supported.')
        

    def forward(self, x1, ref1, x2, ref2):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x1_features, x2_features = self.model(x1), self.model(x2)
        ref1_features, ref2_features = self.model(ref1.detach()), self.model(ref2.detach())
        
        # calculate appearance loss
        if self.rinter_weight > 0:
            inter_change_loss = 0
            for k in self.layer_weights_inter.keys():
                ## calculate delta
                if 'gram' in self.delta_inter_type:
                    inter_change1 = self._gram_mat(x1_features[k]-ref1_features[k]) \
                                    if self.delta_inter_type == 'gram_change' else self._gram_mat(x1_features[k]) - self._gram_mat(ref1_features[k])
                    inter_change2 = self._gram_mat(x2_features[k]-ref2_features[k]) \
                                    if self.delta_inter_type == 'gram_change' else self._gram_mat(x2_features[k]) - self._gram_mat(ref2_features[k])
                else:
                    inter_change1 = self.delta_inter(x1_features[k], ref1_features[k])
                    inter_change2 = self.delta_inter(x2_features[k], ref2_features[k])
                ## calculate loss    
                if self.cri_inter == 'fro':
                    inter_change_loss += torch.norm(inter_change1 - inter_change2, p='fro') * self.layer_weights_inter[k]
                else:
                    inter_change_loss += self.criterion_inter(inter_change1, inter_change2) * self.layer_weights_inter[k]

            inter_change_loss *= self.rinter_weight
        else:
            inter_change_loss = None

        # calculate content loss
        if self.rintra_weight > 0:
            intra_dis_loss = 0
            for k in self.layer_weights_intra.keys():
                #### special case 1
                if self.cri_intra == 'cosine_sim':
                    intra_dis1 = (x1_features[k]- x2_features[k])
                    intra_dis2 = (ref1_features[k]- ref2_features[k])
                    intra_dis_loss += self.criterion_intra(intra_dis1, intra_dis2) * self.layer_weights_intra[k]
                else:
                    ### calculate delta
                    if self.delta_intra_type == 'fro':
                        intra_dis1 = torch.norm(x1_features[k] - x2_features[k], p='fro')
                        intra_dis2 = torch.norm(ref1_features[k] - ref2_features[k], p='fro')
                    else:
                        intra_dis1 = self.delta_intra(x1_features[k], x2_features[k])
                        intra_dis2 = self.delta_intra(ref1_features[k], ref2_features[k])
                    #### calculate loss
                    if self.cri_intra == 'fro':
                        intra_dis_loss += torch.norm(intra_dis1-intra_dis2, p='fro') * self.layer_weights_intra[k]
                    else:
                        intra_dis_loss += self.criterion_intra(intra_dis1, intra_dis2) * self.layer_weights_intra[k]
            intra_dis_loss *= self.rintra_weight
        else:
            intra_dis_loss = None

        return inter_change_loss, intra_dis_loss
    
    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


