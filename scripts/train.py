# Based on https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py

import math
import torch
import random
import shutil
import logging
import os.path
import argparse
import diffusers
import transformers
from pathlib import Path
from tqdm.auto import tqdm
from torch.amp import autocast
from datetime import timedelta
from dataclasses import asdict
from diffusers import LTXPipeline
from torch.utils.data import DataLoader
from typing import List, Optional, Union
from accelerate.logging import get_logger
from core.registry import TransformerConfig
from utils.video.compose import make_video_grid
from diffusers.optimization import get_scheduler
from utils.io.filenames import get_unique_filename
from accelerate.utils import InitProcessGroupKwargs
from accelerate import Accelerator, DistributedType
from utils.text.pocessing import text_preprocessing
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import is_compiled_module
from pipelines.ltxvcondition_v2v import LTXConditionPipeline
from schedulers.frame_window import FrameWindowTimeStepSampler
from schedulers.shifted_logit import ShiftedLogitNormalTimestepSampler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.utils import check_min_version, export_to_video, is_wandb_available, \
    deprecate, convert_unet_state_dict_to_peft
from datasets.raw import VideoDataset, BucketSampler
from datasets.precomputed import PrecomputedDatasetSampleable, \
    PrecomputedDatasetSampleableVal, FixedLengthBatchSampler
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from utils.tensors.latents import _normalize_latents, _pack_latents, _denormalize_latents, _unpack_latents



if is_wandb_available():
    import wandb
    os.environ["WANDB_MODE"] = "offline"

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)

dtype_map = {
    'float32': torch.float32,
    'f32': torch.float32,
    'float16': torch.float16,
    'f16': torch.float16,
    'bfloat16': torch.bfloat16,
    'bf16': torch.bfloat16,
}


def get_args():
    parser = argparse.ArgumentParser(description="Training script for FlowC2S (based on LTXV).")

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--text_encoder_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to text encoder model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default=None,
        help="The path to core dataset parquet/csv file.",
    )

    parser.add_argument(
        "--validation_dataset_path",
        type=str,
        default=None,
        help="The path to validation dataset parquet/csv file.",
    )

    parser.add_argument(
        "--video_init_dataset_root",
        type=str,
        default=None,
        help="Local directory where videos from the initial distribution are stored.",
    )

    parser.add_argument(
        "--video_data_dataset_root",
        type=str,
        default=None,
        help="Local directory where videos from the data distribution are stored.",
    )

    parser.add_argument(
        "--validation_init_video_dataset_root",
        type=str,
        default=None,
        help="Local directory where validation videos from the initial distribution are stored.",
    )

    parser.add_argument(
        "--validation_data_video_dataset_root",
        type=str,
        default=None,
        help="Local directory where validation videos from the data distribution are stored.",
    )


    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--instance_data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Validation
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )

    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )

    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=40,
        help="The number of inference steps to use while sampling validation videos.",
    )

    parser.add_argument(
        "--validation_strength",
        type=float,
        default=2.0,
        help="Strength to use while sampling validation videos.",
    )


    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoders to CPU when they are not used.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--seed_x1", type=int, default=None, help="A seed for reproducible training (data distribution).")

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Denoise in bfloat16",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=257,
        help="Maximum number of frames to use for training.",
    )

    parser.add_argument(
        "--frame_rate",
        type=int,
        default=25,
        help="Frame rate to be used during training and validation.",
    )

    parser.add_argument(
        "--height_buckets",
        type=int,
        nargs="+",
        default=[],
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width_buckets",
        type=int,
        nargs="+",
        default=[],
        help="All input videos are resized to this width.",
    )
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default="center",
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")

    parser.add_argument(
        "--frame_buckets",
        type=int,
        nargs="+",
        default=[],
        help="All input videos will be truncated to these many frames."
    )

    parser.add_argument(
        "--validation_height",
        type=int,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--validation_width",
        type=int,
        help="All input videos are resized to this width.",
    )

    parser.add_argument(
        "--validation_num_frames",
        type=int,
        help="All input videos will be truncated to these many frames."
    )

    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--validation_only_caption",
        action="store_true",
        help="if validation dataset only contains captions",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--validation_batch_size", type=int, default=1, help="Batch size (per device) for the validation dataloader."
    )

    parser.add_argument(
        "--validation_negative_prompt", type=str, default="worst quality, inconsistent motion, blurry, jittery, distorted", help="Negative prompt to be used during validation."
    )

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="",
        help=("The type of the dataset."),
    )

    parser.add_argument(
        "--pretraining_validation",
        action="store_true",
        help=("Whether or not to conduct validation before training."),
    )

    parser.add_argument(
        "--per_token",
        action="store_true",
        help=("Whether to use different sigma levels per token at training."),
    )

    parser.add_argument(
        "--first_frame",
        action="store_true",
        help=("Whether to use small sigma for the first frame."),
    )


    parser.add_argument(
        "--conditioning_p",
        type=float,
        default=0.1,
        help=("Probablity of applying first frame conditioning if first_frame is true."),
    )

    parser.add_argument(
        "--sigma_sampler_type",
        type=str,
        default="ShiftedLogitNormalTimestepSampler",
        help=("The type of sigma sampler for training: can be 'ShiftedLogitNormalTimestepSampler', 'FrameWindowTimeStepSampler'."),
    )

    parser.add_argument(
        "--latent_window_size_for_sigma_sampler",
        type=int,
        default=None,
        help=("Window size in the latent space for FrameWindowTimeStepSampler."),
    )

    parser.add_argument(
        "--mask_free_loss",
        action="store_true",
        help=("Whether to use masked loss function or not."),
    )

    parser.add_argument(
        "--transformer_dtype",
        type=str,
        default="bf16",
        help=("Torch dtype for DiT."),
    )

    parser.add_argument(
        "--dist_regularization_prob",
        type=float,
        default=0.7,
        help=("The probability of applying the inverted latents."),
    )

    parser.add_argument(
        "--training_from_scratch",
        action="store_true",
        help="Whether or not to core from scratch.",
    )

    return parser.parse_args()

@torch.no_grad()
def log_validation(
        pipe,
        args,
        accelerator,
        validation_dataloader,
        global_step,
        is_final_validation: bool = False):
    os.makedirs(f"{args.output_dir}/validation_logs", exist_ok=True)
    validation_logging_dir = f"{args.output_dir}/validation_logs/{global_step}"
    os.makedirs(validation_logging_dir, exist_ok = True)
    logger.info(
        f"Running validation... \n Generating {len(validation_dataloader)} videos."
    )

    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    # fixed validation, different for each video
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    all_prompts = []
    video_filenames = []
    condition_filenames = []
    conditioning_indices = []
    all_apply_inversions = []

    for apply_inversion in [True, False]:
        for batch in tqdm(validation_dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
            prompt = batch["prompts"]
            all_prompts.extend(prompt)
            conditioning_index = torch.tensor([-1])
            conditioning_image = None

            x0_posterior = DiagonalGaussianDistribution(batch["dist_params_p_init"]["parameters"].to(accelerator.device))
            x1_posterior = DiagonalGaussianDistribution(batch["dist_params_p_data"]["parameters"].to(accelerator.device))

            if apply_inversion:
                x1_inverted_latents = batch["p_data_inverted"].to(accelerator.device)
                _, _, latent_num_frames, latent_height, latent_width = x0_posterior.mean.shape
                x1_inverted_latents = _unpack_latents(x1_inverted_latents,
                                                      latent_num_frames,
                                                      latent_height,
                                                      latent_width, 1, 1)
                x1_inverted_latents = _denormalize_latents(x1_inverted_latents,
                                                           pipe.vae.latents_mean,
                                                           pipe.vae.latents_std,
                                                           pipe.vae.config.scaling_factor)
                model_input = x0_posterior.mean + x0_posterior.std * x1_inverted_latents
            else:
                # model_input = x0_posterior.sample(generator=generator) # per-video different seed, the same for the same video
                model_input = x0_posterior.mode()

            all_apply_inversions.append(apply_inversion)

            model_input = model_input.to(pipe.vae.dtype)

            model_input_decoded = pipe.vae.decode(model_input.to(accelerator.device),
                                                  torch.tensor(0.0, device=accelerator.device,
                                                               dtype=model_input.dtype),
                                                  return_dict=False)[0]
            model_input_decoded = pipe.video_processor.postprocess_video(model_input_decoded, output_type="pil")


            gt = x1_posterior.sample(generator=generator)
            gt = gt.to(pipe.vae.dtype)

            gt_decoded = pipe.vae.decode(gt.to(accelerator.device),
                                                  torch.tensor(0.0, device=accelerator.device,
                                                               dtype=gt.dtype),
                                                  return_dict=False)[0]
            gt_decoded = pipe.video_processor.postprocess_video(gt_decoded, output_type="pil")

            model_input = _normalize_latents(model_input,
                                             pipe.vae.latents_mean,
                                             pipe.vae.latents_std)
            model_input = _pack_latents(model_input, 1, 1)
            model_input.squeeze_(1)

            with autocast(accelerator.device.type, dtype=torch.bfloat16):
                video = pipe(
                    latents = model_input,
                    prompt=prompt[0],
                    negative_prompt=args.validation_negative_prompt,
                    num_inference_steps=args.validation_num_inference_steps,
                    num_videos_per_prompt=1,
                    guidance_scale=args.validation_guidance_scale,
                    generator=generator,
                    callback_on_step_end=None,
                    height=args.validation_height,
                    width=args.validation_width,
                    num_frames=args.validation_num_frames,
                    frame_rate=args.frame_rate,
                ).frames[0]

            # convert all to torch tennsor, concat, convert back to pil, save.
            video_grid = make_video_grid([model_input_decoded[0], gt_decoded[0], video], nrow=3)

            base_filename = f"text_to_vid_apply_inversion_{apply_inversion}"
            output_filename = get_unique_filename(
                base_filename,
                ".mp4",
                prompt=prompt[0],
                seed=args.seed,
                resolution=(args.validation_height, args.validation_width, args.validation_num_frames),
                dir=Path(validation_logging_dir),
            )

            output_filename = output_filename.with_name(f"{output_filename.stem}_conditioning_index_{conditioning_index[0].item()}_{output_filename.suffix}")
            export_to_video(video_grid, output_filename, fps=args.frame_rate)
            video_filenames.append(output_filename.as_posix())

            if conditioning_image is not None:
                conditioning_filename = output_filename.with_name(f"{output_filename.stem}.png")
                conditioning_image.save(conditioning_filename)
                condition_filenames.append(conditioning_filename.as_posix())
                conditioning_indices.append(conditioning_index[0].item())

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            log_dict = {}
            log_dict[f"{phase_name}_generated_video"] = [
                        wandb.Video(filename, caption=f"{i}: Apply Inversion: {all_apply_inversions[i]} - {all_prompts[i]}")
                        for i, filename in enumerate(video_filenames)]

            if len(conditioning_indices) > 0:
                log_dict[f"{phase_name}_conditioning_image"] = [
                        wandb.Image(filename, caption=f"conditioning index: {conditioning_indices[i]}")
                        for i, filename in enumerate(condition_filenames)]

            tracker.log(log_dict)

    del pipe
    free_memory()

# copied and modified from pipeline
# Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
@torch.no_grad()
def encode_prompt(
    prompt: Union[str, List[str]],
    clean_caption: bool = False,
    max_sequence_length: int = 128,
    text_encoder: T5Tokenizer = None,
    tokenizer: T5EncoderModel = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
            instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
            This should be "".
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            whether to use classifier free guidance or not
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            number of images that should be generated per prompt
        device: (`torch.device`, *optional*):
            torch device to place the resulting embeddings on
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings.
        clean_caption (bool, defaults to `False`):
            If `True`, the function will preprocess and clean the provided caption before encoding.
    """

    if "mask_feature" in kwargs:
        deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
        deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

    prompt = text_preprocessing(prompt, clean_caption=clean_caption)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(
        prompt, padding="longest", return_tensors="pt"
    ).input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[
        -1
    ] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, max_sequence_length - 1 : -1]
        )

    prompt_attention_mask = text_inputs.attention_mask
    # prompt_attention_mask = prompt_attention_mask.to(device)
    prompt_attention_mask = prompt_attention_mask.bool().to(device)

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_attention_mask = prompt_attention_mask.view(prompt_embeds.shape[0], -1)

    return prompt_embeds, prompt_attention_mask


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # I am setting find_unused_parameters to True, since we core the full model,
    # needs to be changed to False, when experimenting with LoRA
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    kwargs_init = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))  # 30 minutes
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, kwargs_init],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Prepare models and scheduler
    tokenizer = T5Tokenizer.from_pretrained(
        args.text_encoder_model_name_or_path, subfolder="tokenizer"
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )


    if args.training_from_scratch:
        transformer_config = TransformerConfig()
        transformer_config = asdict(transformer_config)
        transformer = LTXVideoTransformer3DModel.from_config(transformer_config,
                                                             torch_dtype=dtype_map.get(args.transformer_dtype.lower()))
    else:
        transformer = LTXVideoTransformer3DModel.from_pretrained(args.pretrained_model_name_or_path,
                                                                 subfolder="transformer",
                                                                 torch_dtype=dtype_map.get(args.transformer_dtype.lower()))

    vae = AutoencoderKLLTXVideo.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",
                                                             torch_dtype=torch.bfloat16)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path,
                                                                subfolder="scheduler")
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    if args.lora_alpha is not None and args.rank is not None:
        transformer.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
                "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
                and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
                "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
                and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # to avoid downcasting and then upcasting
    # will cast to dtype only then doing a lora training
    # otherwise will keep in transformer_dtype (e.g. f32),
    if args.lora_alpha is not None and args.rank is not None:
        transformer.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()


    if args.lora_alpha is not None and args.rank is not None:
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if args.lora_alpha is not None and args.rank is not None:
            if accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(model, type(unwrap_model(transformer))):
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                LTXConditionPipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )
        else:
            if accelerator.is_main_process:
                for model in models:
                    # model.save_pretrained(os.path.join(output_dir, "transformer"))
                    if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                        model = unwrap_model(model)
                        model.save_pretrained(os.path.join(output_dir, "transformer"))
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None
        if args.lora_alpha is not None and args.rank is not None:
            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_ = model
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

            lora_state_dict = LTXConditionPipeline.lora_state_dict(input_dir)

            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if
                k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
        else:
            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_ = model
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

            load_model = LTXVideoTransformer3DModel.from_pretrained(input_dir,
                                                                    subfolder="transformer",
                                                                    torch_dtype=dtype_map.get(args.transformer_dtype.lower()))
            transformer_.register_to_config(**load_model.config)
            transformer_.load_state_dict(load_model.state_dict())

            del load_model
            del model

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    # this can be useful if we decide to fine-tune only a set of parameters
    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    use_deepspeed_optimizer = (
            accelerator.state.deepspeed_plugin is not None
            and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
            accelerator.state.deepspeed_plugin is not None
            and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    video_resolution_buckets = []
    for frames in args.frame_buckets:
        for height in args.height_buckets:
            for width in args.width_buckets:
                video_resolution_buckets.append((frames, height, width))

    # Dataset and DataLoader
    if args.dataset_type == "precomputed":
        train_dataset = PrecomputedDatasetSampleable(args.video_init_dataset_root,
                                                     args.video_data_dataset_root,
                                                     shuffle_init=False,
                                                     load_inverted_latents=True)
    else:
        train_dataset = VideoDataset(
            dataset_path=args.train_dataset_path,
            video_dataset_root=args.video_dataset_root,
            caption_column=args.caption_column,
            video_column=args.video_column,
            video_reshape_mode=args.video_reshape_mode,
            fps=args.fps,
            max_num_frames=args.max_num_frames,
            skip_frames_start=args.skip_frames_start,
            skip_frames_end=args.skip_frames_end,
            cache_dir=args.cache_dir,
            id_token=args.id_token,
            resolution_buckets=video_resolution_buckets,
            shuffle_df=True,
            validation=False,
        )

    validation_dataset = PrecomputedDatasetSampleableVal(
        [args.validation_init_video_dataset_root],
        [args.validation_data_video_dataset_root],
        args.validation_dataset_path, args.video_column, args.caption_column,
        load_inverted_latents=True)

    @torch.no_grad()
    def encode_video(video, conditioning_index, video_len, vae):
        latent_sample = vae.encode(video).latent_dist.sample()
        latent_sample = _normalize_latents(latent_sample, vae.latents_mean, vae.latents_std)

        return latent_sample

    def collate_fn(examples):
        # videos = [example["instance_video"][None] for example in examples]
        # prompts = [example["instance_prompt"] for example in examples]

        prompts = [example["instance_prompt"] for example in examples[0]]
        videos = [example["instance_video"][None] for example in examples[0]]

        videos = torch.cat(videos)  # [B, F, C, H, W]
        videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        videos = videos.to(memory_format=torch.contiguous_format).float()
        output_dict = {
            "videos": videos,
            "prompts": prompts,
        }
        return output_dict

    if args.dataset_type == "precomputed":
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_num_workers>0,
            batch_sampler=FixedLengthBatchSampler(train_dataset, batch_size=args.train_batch_size,
                                            shuffle=True, drop_last=False, random_state=args.seed),
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_num_workers>0,
            sampler=BucketSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True),
        )


    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_num_workers > 0,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    frame_rate = 24
    latent_frame_rate = frame_rate / 8
    spatial_compression_ratio = vae.spatial_compression_ratio
    rope_interpolation_scale = [1 / latent_frame_rate, spatial_compression_ratio, spatial_compression_ratio]

    transformer_spatial_patch_size = transformer.config.patch_size
    transformer_temporal_patch_size = (
        transformer.config.patch_size_t
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "Exp1-ltx-streaming"
        accelerator.init_trackers(tracker_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total core batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Distributed type = {accelerator.distributed_type}")
    logger.info(f"  Transformer type = {unwrap_model(transformer).dtype}")
    logger.info(f"  Training From Scratch = {args.training_from_scratch}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if args.resume_from_checkpoint != "latest":
                accelerator.load_state(path)
                global_step = int(path.split("/")[-2].split("-")[1])
            else:
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    if args.sigma_sampler_type == "ShiftedLogitNormalTimestepSampler":
        sigma_sampler = ShiftedLogitNormalTimestepSampler(per_token=args.per_token, conditioning_p=args.conditioning_p)
    elif args.sigma_sampler_type == "FrameWindowTimeStepSampler":
        assert args.latent_window_size_for_sigma_sampler is not None
        sigma_sampler = FrameWindowTimeStepSampler(per_token=args.per_token, conditioning_p=args.conditioning_p,
                                                   latent_window_size=args.latent_window_size_for_sigma_sampler)
    else:
        raise ValueError(f"Unknown sigma_sampler: {args.sigma_sampler}")

    if args.pretraining_validation:
        if accelerator.is_main_process:
            # Create pipeline
            if args.validation_only_caption:
                pipe = LTXPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                   transformer=unwrap_model(transformer),
                                   vae=vae,
                                   text_encoder=text_encoder,
                                   tokenizer=tokenizer,
                                   scheduler=scheduler)
            else:
                pipe = LTXConditionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    scheduler=scheduler)

            log_validation(
                pipe=pipe,
                args=args,
                accelerator=accelerator,
                validation_dataloader=validation_dataloader,
                global_step=0,
            )

    if args.offload:
        vae.cpu()
        text_encoder.cpu()

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        generator_x1 = torch.Generator(device=accelerator.device).manual_seed(args.seed_x1)
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                if args.dataset_type == "precomputed":
                    x0_posterior = DiagonalGaussianDistribution(batch["dist_params_p_init"]["parameters"])

                    random_random = random.random()
                    if random_random < args.dist_regularization_prob:
                        x1_inverted_latents =  batch["p_data_inverted"]
                        _, _, latent_num_frames, latent_height, latent_width = x0_posterior.mean.shape
                        x1_inverted_latents = _unpack_latents(x1_inverted_latents,
                                                              latent_num_frames,
                                                              latent_height,
                                                              latent_width,
                                                              1, 1)
                        x1_inverted_latents = _denormalize_latents(x1_inverted_latents,
                                                                   vae.latents_mean,
                                                                   vae.latents_std,
                                                                   vae.config.scaling_factor)
                        x0 = x0_posterior.mean + x0_posterior.std * x1_inverted_latents
                    else:
                        x0 = x0_posterior.mode()

                    x0 = _normalize_latents(x0, vae.latents_mean, vae.latents_std)
                    x0 = _pack_latents(x0, 1, 1)
                    x0.squeeze_(1)

                    x1_posterior = DiagonalGaussianDistribution(batch["dist_params_p_data"]["parameters"])
                    # fixing the generator per epoch so as the video has the same latent code when being encountered
                    # for the second time during the training
                    latent_conditions = x1_posterior.sample(generator=generator_x1)
                    _, _, latent_num_frames, latent_height, latent_width = latent_conditions.shape
                    latent_conditions = _normalize_latents(latent_conditions, vae.latents_mean, vae.latents_std)
                    model_input = _pack_latents(latent_conditions, 1, 1)
                    model_input.squeeze_(1)
                else:
                    # [B, C, F, H, W]
                    model_input = batch["videos"]
                    video_len = batch["videos"].shape[2]
                    conditioning_index = torch.tensor([0])

                    # accelerator.print(f"model_input shape: {model_input.shape}")
                    model_input = encode_video(model_input.to(accelerator.device,
                                                                  dtype=weight_dtype),
                                               conditioning_index, video_len,
                                               vae.to(accelerator.device))
                    latent_num_frames, latent_height, latent_width = model_input.shape[2:]
                    model_input = _pack_latents(model_input, transformer_spatial_patch_size, transformer_temporal_patch_size)


                if args.dataset_type == "precomputed":
                    text_conditions = batch["text_conditions"]
                    prompt_embeds = text_conditions["prompt_embeds"].to(weight_dtype)
                    prompt_attention_mask = text_conditions["prompt_attention_mask"].to(weight_dtype)
                else:
                    prompts = batch["prompts"]
                    # encode prompts
                    prompt_embeds, prompt_attention_mask = encode_prompt(
                        tokenizer=tokenizer,
                        text_encoder=text_encoder.to(accelerator.device),
                        prompt=prompts,
                        max_sequence_length=128, # hard-coded from now - from LTXPipeline
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )

                # if args.offload:
                #     text_encoder.cpu()

                sigmas = sigma_sampler.sample_for(model_input, latent_num_frames)
                sigmas = sigmas.unsqueeze(-1)
                timesteps = (sigmas * scheduler.config.num_train_timesteps).long()

                if not args.mask_free_loss:
                    loss_mask = torch.ones_like(model_input)

                if args.per_token:
                    if not args.mask_free_loss:
                        loss_mask = sigmas.clone()
                elif args.first_frame:
                    if args.conditioning_p and random.random() < args.conditioning_p:
                        sigmas = sigmas.repeat(1, model_input.shape[1], 1)
                        first_frame_end_idx = latent_height * latent_width
                        sigmas[:, :first_frame_end_idx] = 1e-5  # Small sigma close to 0 for the first frame.
                        if not args.mask_free_loss:
                            loss_mask[:, :first_frame_end_idx] = 0.0
                else:
                    pass

                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * x0

                # Predict the velocity (FlowMatching)
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    rope_interpolation_scale=rope_interpolation_scale,
                    # video_coords=video_ids,
                    encoder_attention_mask=prompt_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    return_dict=False,
                )[0]

                target = x0 - model_input
                loss = (model_output - target) ** 2
                if not args.mask_free_loss:
                    loss = loss.mul(loss_mask).div(loss_mask.mean())
                    loss = loss.mean()
                else:
                    loss = loss.mean()
                accelerator.backward(loss)

                torch.cuda.empty_cache()

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0:
                        if args.validation_only_caption:
                            pipe = LTXPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                               transformer=unwrap_model(transformer),
                                                               vae=vae,
                                                               text_encoder=text_encoder,
                                                               tokenizer=tokenizer,
                                                               scheduler=scheduler)
                        else:
                            pipe = LTXConditionPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                transformer=unwrap_model(transformer),
                                vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                scheduler=scheduler)

                        log_validation(
                            pipe=pipe,
                            args=args,
                            accelerator=accelerator,
                            validation_dataloader=validation_dataloader,
                            global_step=global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)

        if args.lora_alpha is not None and args.rank is not None:
            transformer_lora_layers = get_peft_model_state_dict(transformer)

            LTXConditionPipeline.save_lora_weights(
                save_directory=args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
            )
        else:
            transformer.save_pretrained(args.output_dir)

        # Cleanup trained models to save memory
        del transformer
        free_memory()

        if args.lora_alpha is not None and args.rank is not None:
            transformer = LTXVideoTransformer3DModel.from_pretrained(args.pretrained_model_name_or_path,
                                                                     subfolder="transformer",
                                                                     torch_dtype=dtype_map.get(args.transformer_dtype.lower()))
        else:
            transformer = LTXVideoTransformer3DModel.from_pretrained(args.output_dir,
                                                                     subfolder="transformer",
                                                                     torch_dtype=dtype_map.get(args.transformer_dtype.lower()))

        vae = AutoencoderKLLTXVideo.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",
                                                    torch_dtype=torch.bfloat16)

        if args.validation_only_caption:
            pipe = LTXPipeline.from_pretrained(args.pretrained_model_name_or_path,
                               transformer=transformer,
                               vae=vae)
        else:
            pipe = LTXConditionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,
                vae=vae)

        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

        if args.lora_alpha is not None and args.rank is not None:
            lora_scaling = args.lora_alpha / args.rank
            pipe.load_lora_weights(args.output_dir, adapter_name="ltxv-lora")
            pipe.set_adapters(["ltxv-lora"], [lora_scaling])

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()

        # Run inference
        log_validation(
            pipe=pipe,
            args=args,
            accelerator=accelerator,
            validation_dataloader=validation_dataloader,
            global_step=global_step,
            is_final_validation=True,
        )

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
