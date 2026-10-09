"""
Module: models.py
Description: Defines various student architectures for Knowledge Distillation (KD).
             All models are based on the ResidualEncoderUNet with varying capacities.
"""

import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

def get_large_student() -> ResidualEncoderUNet:
    """
    Returns a large-capacity student model (~50M Parameters).
    """
    model = ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(28, 56, 112, 224, 320, 320),
        conv_op=nn.Conv3d, 
        kernel_sizes=[[3,3,3]]*6, 
        strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
        n_blocks_per_stage=(1, 2, 3, 3, 3, 3),
        num_classes=2, 
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True, 
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs={'inplace': True}, 
        deep_supervision=False
    )
    return model

def get_medium_student() -> ResidualEncoderUNet:
    """
    Returns a medium-capacity student model (~35M Parameters).
    Slightly shallower than the baseline teacher.
    """
    model = ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(24, 48, 96, 192, 256, 256), 
        conv_op=nn.Conv3d, 
        kernel_sizes=[[3,3,3]]*6, 
        strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
        n_blocks_per_stage=(1, 2, 3, 3, 3, 3),
        num_classes=2, 
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True, 
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs={'inplace': True}, 
        deep_supervision=False
    )
    return model

def get_small_student() -> ResidualEncoderUNet:
    """
    Returns a small-capacity student model (~17M Parameters).
    """
    model = ResidualEncoderUNet(
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
    return model

def get_light_student() -> ResidualEncoderUNet:
    """
    Returns a light-capacity student model (~10M Parameters).
    """
    model = ResidualEncoderUNet(
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
    return model

def get_extra_light_student() -> ResidualEncoderUNet:
    """
    Returns an extra-light-capacity student model (~5M Parameters).
    """
    model = ResidualEncoderUNet(
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
    return model

def get_extra_extralight_student() -> ResidualEncoderUNet:
    """
    Returns an extra-extra-light-capacity student model (~2.5M Parameters).
    """
    model = ResidualEncoderUNet(
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
    return model

def get_nano_student() -> ResidualEncoderUNet:
    """
    Returns a nano-capacity student model (~0.5M - 1M Parameters).
    Designed to test the absolute lower bounds of capacity.
    """
    model = ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(4, 8, 16, 32, 48, 48), 
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
    return model

def get_pico_student() -> ResidualEncoderUNet:
    """
    Returns a pico-capacity student model (~100K - 150K Parameters).
    Strictly smaller than Nano. Drops base channels to 2.
    """
    model = ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(2, 4, 8, 16, 24, 24),
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
    return model

def get_femto_student() -> ResidualEncoderUNet:
    """
    Returns a femto-capacity student model (~30K - 40K Parameters).
    An absolute extreme capacity test. The first layer operates with only 1 filter.
    """
    model = ResidualEncoderUNet(
        input_channels=1, 
        n_stages=6, 
        features_per_stage=(1, 2, 4, 8, 12, 12), 
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
    return model

def get_halved_student() -> ResidualEncoderUNet:
    """
    Returns a halved-capacity Residual Encoder UNet.
    """
    model = ResidualEncoderUNet(
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
    return model