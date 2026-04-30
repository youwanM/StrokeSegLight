import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

def get_large_student():
    """
    ~ 50M Parameters.
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
    ~~ 35M Parameters.
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
    ~ 17M Parameters.
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
    ~ 10M Parameters.
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

def get_extra_light_student():
    """
    ~ 5M Parameters.
    """
    return ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(12, 24, 48, 80, 96, 128), 
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

def get_extra_extralight_student():
    """
    ~ 2.5M Parameters.
    """
    return ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(8, 16, 32, 64, 80, 80), 
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