import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

def get_large_student():
    """
    ~ 50M Parameters.
    Un modèle intermédiaire supérieur, se situant entre le Medium (~35M) et le Teacher.
    On augmente la capacité (largeur) en démarrant à 28 canaux.
    """
    return ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(28, 56, 112, 224, 320, 320), # Augmentation des canaux
        conv_op=nn.Conv3d, 
        kernel_sizes=[[3,3,3]]*6, 
        strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
        n_blocks_per_stage=(1, 2, 3, 3, 3, 3), # On conserve la même profondeur que le Medium
        num_classes=2, 
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True, 
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs={'inplace': True}, 
        deep_supervision=False
    )

def get_medium_student():
    """
    ~ Factor 2 Reduction (50% Smaller than Teacher).
    We reduce the channels slightly and keep the network relatively deep.
    """
    return ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(24, 48, 96, 192, 256, 256), 
        conv_op=nn.Conv3d, 
        kernel_sizes=[[3,3,3]]*6, 
        strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
        n_blocks_per_stage=(1, 2, 3, 3, 3, 3), # Slightly shallower than baseline
        num_classes=2, 
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True, 
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs={'inplace': True}, 
        deep_supervision=False
    )

def get_small_student():
    """
    ~ Factor 5 Reduction (80% Smaller than Teacher).
    A great middle ground between the Light and Medium students.
    """
    return ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(20, 40, 80, 160, 200, 200), 
        conv_op=nn.Conv3d, 
        kernel_sizes=[[3,3,3]]*6, 
        strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
        n_blocks_per_stage=(1, 2, 2, 2, 2, 2), 
        num_classes=2, 
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True, 
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs={'inplace': True}, 
        deep_supervision=False
    )

def get_light_student():
    """
    ~ Factor 10 Reduction (90% Smaller than Teacher)
    """
    return ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(16, 32, 64, 128, 160, 160), 
        conv_op=nn.Conv3d, 
        kernel_sizes=[[3,3,3]]*6, 
        strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
        n_blocks_per_stage=(1, 1, 2, 2, 2, 2), 
        num_classes=2, 
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True, 
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs={'inplace': True}, 
        deep_supervision=False
    )


    """
    Halved-capacity Residual Encoder UNet.
    """
    return ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(16, 32, 64, 128, 160, 160), # Halved Features
        conv_op=nn.Conv3d, 
        kernel_sizes=[[3,3,3]]*6, 
        strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
        n_blocks_per_stage=(1, 1, 2, 2, 2, 2), 
        num_classes=2, 
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True, 
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs={'inplace': True}, 
        deep_supervision=False
    )