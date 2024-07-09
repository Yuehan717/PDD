from torch import nn as nn
from basicsr.archs.arch_util import DWT_Haar
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch


@ARCH_REGISTRY.register()
class MlpDiscriminator(nn.Module):
    """Mlp discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True, sigmoid=True):
        super(MlpDiscriminator, self).__init__()
        if sigmoid:
            final_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)
        if batch_norm:
            self.body = nn.Sequential(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                final_layer
            )
        else:
            self.body=nn.Sequential(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                final_layer
            )
    def forward(self, x):
        return self.body(x)


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 128 or self.input_size == 256 or self.input_size == 192, (
            f'input size must be 128, 192 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        if self.input_size == 192:
            self.linear1 = nn.Linear(num_feat * 8 * 6 * 6, 100)
        else:
            self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
            
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

@ARCH_REGISTRY.register()
class UNetDiscriminatorWithSpectralNorm(nn.Module):
    """A U-Net discriminator with spectral normalization.

    Args:
        num_in_ch (int): Channel number of the input.
        num_feat (int, optional): Channel number of the intermediate
            features. Default: 64.
        skip_connection (bool, optional): Whether to use skip connection.
            Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):

        super(UNetDiscriminatorWithSpectralNorm, self).__init__()

        self.skip_connection = skip_connection

        self.conv_0 = nn.Conv2d(
            num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # downsample
        self.conv_1 = spectral_norm(
            nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv_2 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv_3 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # upsample
        self.conv_4 = spectral_norm(
            nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv_5 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv_6 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # final layers
        self.conv_7 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv_8 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv_9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        """Forward function.

        Args:
            img (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat_0 = self.lrelu(self.conv_0(img))

        # downsample
        feat_1 = self.lrelu(self.conv_1(feat_0))
        feat_2 = self.lrelu(self.conv_2(feat_1))
        feat_3 = self.lrelu(self.conv_3(feat_2))

        # upsample
        feat_3 = self.upsample(feat_3)
        feat_4 = self.lrelu(self.conv_4(feat_3))
        if self.skip_connection:
            feat_4 = feat_4 + feat_2

        feat_4 = self.upsample(feat_4)
        feat_5 = self.lrelu(self.conv_5(feat_4))
        if self.skip_connection:
            feat_5 = feat_5 + feat_1

        feat_5 = self.upsample(feat_5)
        feat_6 = self.lrelu(self.conv_6(feat_5))
        if self.skip_connection:
            feat_6 = feat_6 + feat_0

        # final layers
        out = self.lrelu(self.conv_7(feat_6))
        out = self.lrelu(self.conv_8(out))

        return self.conv_9(out)

    # def init_weights(self, pretrained=None, strict=True):
    #     """Init weights for models.

    #     Args:
    #         pretrained (str, optional): Path for pretrained weights. If given
    #             None, pretrained weights will not be loaded. Defaults to None.
    #         strict (boo, optional): Whether strictly load the pretrained model.
    #             Defaults to True.
    #     """

    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=strict, logger=logger)
    #     elif pretrained is not None:  # Use PyTorch default initialization.
    #         raise TypeError(f'"pretrained" must be a str or None. '
    #                         f'But received {type(pretrained)}.')


@ARCH_REGISTRY.register()
class PatchDiscriminatorWithSpectralNorm(nn.Module):
    """A U-Net discriminator with spectral normalization.

    Args:
        num_in_ch (int): Channel number of the input.
        num_feat (int, optional): Channel number of the intermediate
            features. Default: 64.
        skip_connection (bool, optional): Whether to use skip connection.
            Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):

        super(PatchDiscriminatorWithSpectralNorm, self).__init__()

        self.skip_connection = skip_connection

        self.conv_0 = nn.Conv2d(
            num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # downsample
        self.conv_1 = spectral_norm(
            nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv_2 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv_3 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # upsample
        self.conv_4 = spectral_norm(
            nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv_5 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv_6 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # final layers
        self.conv_7 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv_8 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv_9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        """Forward function.

        Args:
            img (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat_0 = self.lrelu(self.conv_0(img))

        # downsample
        feat_1 = self.lrelu(self.conv_1(feat_0))
        feat_2 = self.lrelu(self.conv_2(feat_1))
        feat_3 = self.lrelu(self.conv_3(feat_2))

        # upsample
        feat_3 = self.upsample(feat_3)
        feat_4 = self.lrelu(self.conv_4(feat_3))
        if self.skip_connection:
            feat_4 = feat_4 + feat_2

        feat_4 = self.upsample(feat_4)
        feat_5 = self.lrelu(self.conv_5(feat_4))
        if self.skip_connection:
            feat_5 = feat_5 + feat_1

        feat_5 = self.upsample(feat_5)
        feat_6 = self.lrelu(self.conv_6(feat_5))
        if self.skip_connection:
            feat_6 = feat_6 + feat_0

        # final layers
        out = self.lrelu(self.conv_7(feat_6))
        out = self.lrelu(self.conv_8(out))

        return self.conv_9(out)
