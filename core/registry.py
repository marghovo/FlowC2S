import torch
from typing import Type, Dict
from dataclasses import dataclass
from pipelines.ltxvcondition_v2v import LTXConditionPipeline
from diffusers import (AutoencoderKLLTXVideo, LTXVideoTransformer3DModel,
                       FlowMatchEulerDiscreteScheduler, LTXPipeline,
                       WanTransformer3DModel, AutoencoderKLWan,
                       UniPCMultistepScheduler, WanPipeline)
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, UMT5EncoderModel


@dataclass
class LTXVModel:
    transformer: tuple[Type, Dict] = (LTXVideoTransformer3DModel, {
        "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
        "subfolder": "transformer",
        "torch_dtype": torch.bfloat16
    })
    vae:  tuple[Type, Dict] = (AutoencoderKLLTXVideo, {
        "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
        "subfolder": "vae",
        "torch_dtype": torch.bfloat16
    })
    text_encoder:  tuple[Type, Dict] = (T5EncoderModel, {
        "pretrained_model_name_or_path": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "subfolder": "text_encoder",
        "torch_dtype": None
    })
    tokenizer:  tuple[Type, Dict] = (T5Tokenizer, {
        "pretrained_model_name_or_path": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "subfolder": "tokenizer",
    })
    scheduler:  tuple[Type, Dict] = (FlowMatchEulerDiscreteScheduler, {
        "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
        "subfolder": "scheduler",
    })

@dataclass
class LTXVPipeline:
    text_to_video: tuple[Type, Dict] = (LTXPipeline, {
        "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
        "torch_dtype": torch.bfloat16
    })
    image_to_video: tuple[Type, Dict] = (LTXConditionPipeline, {
        "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
        "torch_dtype": torch.bfloat16
    })


@dataclass
class WanModel:
    transformer: tuple[Type, Dict] = (WanTransformer3DModel, {
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "subfolder": "transformer",
        "torch_dtype": torch.float32
    })
    vae:  tuple[Type, Dict] = (AutoencoderKLWan, {
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "subfolder": "vae",
        "torch_dtype": torch.float32
    })
    text_encoder:  tuple[Type, Dict] = (UMT5EncoderModel, {
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "subfolder": "text_encoder",
        "torch_dtype": torch.bfloat16
    })
    tokenizer:  tuple[Type, Dict] = (AutoTokenizer, {
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "subfolder": "tokenizer",
    })
    scheduler:  tuple[Type, Dict] = (UniPCMultistepScheduler, {
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "subfolder": "scheduler",
    })

@dataclass
class WanPipeline:
    text_to_video: tuple[Type, Dict] = (WanPipeline, {
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "torch_dtype": torch.bfloat16
    })


@dataclass
class MixedWANLTXVModel:
    transformer: tuple[Type, Dict] = (WanTransformer3DModel, {
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "subfolder": "transformer",
        "torch_dtype": torch.float32
    })
    vae:  tuple[Type, Dict] = (AutoencoderKLLTXVideo, {
        "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
        "subfolder": "vae",
        "torch_dtype": torch.bfloat16
    })
    text_encoder:  tuple[Type, Dict] = (T5EncoderModel, {
        "pretrained_model_name_or_path": "PixArt-alpha/PixArt-XL-2-1024-MS",
        "subfolder": "text_encoder",
        "torch_dtype": None
    })
    tokenizer:  tuple[Type, Dict] = (AutoTokenizer, {
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "subfolder": "tokenizer",
    })
    scheduler:  tuple[Type, Dict] = (UniPCMultistepScheduler, {
        "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "subfolder": "scheduler",
    })

@dataclass
class TransformerConfig:
    in_channels: int = 128
    out_channels: int = 128
    patch_size: int = 1
    patch_size_t: int = 1
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    cross_attention_dim: int = 2048
    num_layers: int = 28
    activation_fn: str = "gelu-approximate"
    qk_norm: str = "rms_norm_across_heads"
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-06
    caption_channels: int = 4096
    attention_bias: bool = True
    attention_out_bias: bool = True
    _class_name: str = 'LTXVideoTransformer3DModel'
    _diffusers_version: str = '0.33.0.dev0'
