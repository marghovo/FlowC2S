import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from .registry import register
import torch.nn.functional as F
from diffusers import LTXPipeline
from diffusers.utils import load_video
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from diffusers.utils import export_to_video
from utils.video.compose import make_video_grid
from typing import Tuple, Union, Optional, Dict
from utils.tensors.latents import _normalize_latents_wan
from datasets.paths import DatasetVal, DatasetNuscencesVal
from torchvision.transforms.functional import to_pil_image
from utils.tensors.decoding import decode_standalone, decode_standalone_wan
from diffusers import LTXVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from utils.io.filenames import get_unique_filename, convert_prompt_to_filename
from diffusers import WanPipeline, WanTransformer3DModel, UniPCMultistepScheduler
from utils.loading.video import resize_and_load as load_image_to_tensor_with_resize_and_crop


@dataclass
class Context:
    height: Optional[int] = field(default=512, metadata={"args": ["--height"]})
    width: Optional[int] = field(default=768, metadata={"args": ["--width"]})
    downsample_factor: Optional[int] = field(default=1, metadata={"args": ["--downsample_factor"]})
    seed: int = 12321321
    frame_rate: int = field(default=25, metadata={"args": ["--frame_rate"]})
    random_seeds: bool = field(
        default=False,
        metadata={"args": ["--random_seeds"], "action": "store_true"}
    )
    num_inference_steps: int = field(default=40, metadata={"args": ["--num_inference_steps"]})
    num_frames: int = field(default=41, metadata={"args": ["--num_frames"]})
    guidance_scale: float = field(default=3.5, metadata={"args": ["--cfg"]})
    exp_name: str = field(default="exp-name", metadata={"args": ["--exp_name"]})
    logging_dir: str = field(default="exp", metadata={"args": ["--logging_dir"]})
    negative_prompt: Optional[str] = "worst quality, inconsistent motion, blurry, jittery, distorted"
    batch_size: int = field(default=1, metadata={"args": ["--batch_size"]})
    sample_input_latent: bool = field(default=False, metadata={"args": ["--sample_input_latent"]})
    save_full_video: bool = field(default=False, metadata={"args": ["--save_full_video"]})
    save_grid: bool = field(default=False, metadata={"args": ["--save_grid"]})
    individual_videos: bool = field(default=False, metadata={"args": ["--individual_videos"]})
    save_npy: bool = field(default=False, metadata={"args": ["--save_npy"]})
    save_caption: bool = field(default=False, metadata={"args": ["--save_caption"]})
    max_num_of_generated_videos: int = field(default=2001, metadata={"args": ["--max_num_of_generated_videos"]})
    starting_idx: int = field(default=0, metadata={"args": ["--starting_idx"]})


@register("InferencePipeline")
class InferencePipeline:
    def __init__(self, device: str = "cuda") -> None:
        self.pipe = None
        self.device = device

    def load_pipeline(self,
                      pretrained_model_name_or_path: str,
                      transformer_path: str) -> None:
        transformer = LTXVideoTransformer3DModel.from_pretrained(transformer_path,
                                                                 subfolder="transformer",
                                                                 torch_dtype=torch.bfloat16)

        self.pipe = LTXPipeline.from_pretrained(pretrained_model_name_or_path,
                                                transformer=transformer,
                                                torch_dtype=torch.bfloat16)
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def load_lora(self, lora_path: str = None, alpha: int = 32, rank: int = 32) -> None:
        self.lora_path = lora_path
        if lora_path is not None:
            lora_scaling = alpha / rank
            self.pipe.load_lora_weights(lora_path, adapter_name="ltxv-lora")
            self.pipe.set_adapters(["ltxv-lora"], [lora_scaling])

    def set_ctx(self, ctx: Context) -> None:
        self.ctx = ctx

    def load_data(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def infer(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def measure_generation_memory(self, batch) -> (float, float, int):
        total_mem = torch.cuda.memory_allocated()  # bytes
        print(f"Total GPU memory used in measurer generation memory: {total_mem / (1024 ** 3):.3f} GB")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.pipe.to(self.device)

        generator = torch.Generator(device=self.device).manual_seed(self.ctx.seed) if self.ctx.seed else None
        prompt = batch["prompts"][0]
        video_path = batch["video_paths"][0]
        video_tensor, gt_tensor, model_inputs = self.prepare_model_input(video_path)
        effective_volume = model_inputs["model_input"].numel() * 2

        batch_size = model_inputs["model_input"].shape[0]
        latent_num_frames = (self.ctx.num_frames - 1) // self.pipe.vae.temporal_compression_ratio + 1
        latent_height = self.ctx.height // self.pipe.vae.spatial_compression_ratio
        latent_width = self.ctx.width // self.pipe.vae.spatial_compression_ratio
        generation_kwargs = {
            "latents": model_inputs["model_input"],
            "prompt": prompt,
            "negative_prompt": self.ctx.negative_prompt,
            "num_inference_steps": self.ctx.num_inference_steps,
            "num_videos_per_prompt": 1,
            "guidance_scale": self.ctx.guidance_scale,
            "generator": generator,
            "callback_on_step_end": None,
            "height": self.ctx.height,
            "width": self.ctx.width,
            "num_frames": self.ctx.num_frames,
            "frame_rate": self.ctx.frame_rate,
            "output_type": "latent",
        }
        torch.cuda.synchronize()

        # Warm-up
        _ = self.pipe(**generation_kwargs)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated()

        # Generation
        latents = self.pipe(**generation_kwargs).frames

        decode_standalone(
            self.pipe,
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            torch.device(self.device),
            0.0,
            batch_size,
            None,
            "pil",
            generator,
            model_inputs["model_input"].dtype
        )
        torch.cuda.synchronize()

        peak_mem = torch.cuda.max_memory_allocated()
        overhead_MB_VAE = (peak_mem - baseline_mem) / (1024 ** 2)
        overhead_MB = None

        return overhead_MB, overhead_MB_VAE, effective_volume


@register("RawDataInferencePipeline")
class RawDataInferencePipeline(InferencePipeline):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)

    def load_data(self, dataset_path, video_column, caption_column) -> None:
        dataset = DatasetVal(dataset_path, video_column, caption_column)
        self.dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)

    @torch.no_grad()
    def prepare_model_input(self, video_path: Union[str, list[str]]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        video = load_video(video_path) if isinstance(video_path, str) else video_path
        video_tensor = [load_image_to_tensor_with_resize_and_crop(item, self.ctx.height, self.ctx.width) for item in
                        video]
        video_tensor = torch.cat(video_tensor, dim=2)
        video_tensor_input = video_tensor[:, :, :self.ctx.num_frames, :, :]
        _, _, self.num_frames, self.height, self.width = video_tensor_input.shape

        if self.ctx.sample_input_latent:
            latents = self.pipe.vae.encode(
                video_tensor_input.to(dtype=self.pipe.vae.dtype, device=self.device)).latent_dist.sample()
        else:
            latents = self.pipe.vae.encode(
                video_tensor_input.to(dtype=self.pipe.vae.dtype, device=self.device)).latent_dist.mode()
        latents = self.pipe._normalize_latents(latents, self.pipe.vae.latents_mean, self.pipe.vae.latents_std)
        model_input = self.pipe._pack_latents(latents,
                                              self.pipe.transformer_spatial_patch_size,
                                              self.pipe.transformer_temporal_patch_size)

        if self.ctx.batch_size > 1:
            model_input = model_input.repeat(self.ctx.batch_size, 1, 1)

        model_inputs = {
            "model_input": model_input
        }

        return video_tensor_input, video_tensor[:, :, self.ctx.num_frames:2 * self.ctx.num_frames], model_inputs

    @torch.no_grad()
    def get_memory_usage(self, num_runs: int) -> None:
        if self.ctx.random_seeds:
            self.ctx.seed = random.randint(0, 1000000000)

        assert len(self.dataloader) == 1

        memory_logs = []
        for i in range(num_runs):
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Validation")):
                if batch_idx < self.ctx.starting_idx:
                    continue

                if batch_idx >= self.ctx.max_num_of_generated_videos:
                    break
                mem_usage, mem_usage_vae, effective_volume = self.measure_generation_memory(batch)
                memory_logs.append({
                    "Run": i + 1,
                    "Memory_Usage_MB": mem_usage,
                    "Memory_Usage_VAE_MB": mem_usage_vae
                })

        df = pd.DataFrame(memory_logs)
        avg_row = pd.DataFrame([{
            'Run': 'Avg',
            'Memory_Usage_MB': df['Memory_Usage_MB'].mean(),
            'Memory_Usage_VAE_MB': df['Memory_Usage_VAE_MB'].mean()
        }])
        df = pd.concat([df, avg_row], ignore_index=True)
        df.reset_index(inplace=True, drop=True)

        os.makedirs(self.ctx.logging_dir, exist_ok=True)
        df.to_csv(f"{self.ctx.logging_dir}/memory_usage_num_frames_{effective_volume}.csv", index=False)

    @torch.no_grad()
    def infer(self) -> None:
        self.pipe.to(self.device)

        if self.ctx.random_seeds:
            self.ctx.seed = random.randint(0, 1000000000)

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Validation")):
            if batch_idx < self.ctx.starting_idx:
                continue

            if batch_idx >= self.ctx.max_num_of_generated_videos:
                break

            generator = torch.Generator(device=self.device).manual_seed(self.ctx.seed) if self.ctx.seed else None
            prompt = batch["prompts"][0]
            video_path = batch["video_paths"][0]
            if type(video_path) is str:
                video_name = video_path.split("/")[-1].split(".mp4")[0]
            elif type(video_path) is list:
                video_name = video_path[0].split("/")[-1].split(".jpg")[0]
            else:
                video_name = "the_nameless"

            video_tensor, gt_tensor, model_inputs = self.prepare_model_input(video_path)
            output_video = self.pipe(
                latents=model_inputs["model_input"],
                prompt=prompt,
                negative_prompt=self.ctx.negative_prompt,
                num_inference_steps=self.ctx.num_inference_steps,
                num_videos_per_prompt=1,
                guidance_scale=self.ctx.guidance_scale,
                generator=generator,
                callback_on_step_end=None,
                height=self.ctx.height,
                width=self.ctx.width,
                num_frames=self.ctx.num_frames,
                frame_rate=self.ctx.frame_rate,
            ).frames[0]

            video_tensor_denormalized = (video_tensor * 0.5 + 0.5).clamp(0, 1)
            video_tensor_denormalized = video_tensor_denormalized.permute(0, 2, 1, 3, 4)
            video = [to_pil_image(frame) for frame in video_tensor_denormalized[0]]

            gt_denormalized = (gt_tensor * 0.5 + 0.5).clamp(0, 1)
            gt_tensor_denormalized = gt_denormalized.permute(0, 2, 1, 3, 4)
            gt = [to_pil_image(frame) for frame in gt_tensor_denormalized[0]]

            w, h = output_video[0].size
            if self.ctx.downsample_factor > 1:
                w, h = w // self.ctx.downsample_factor, h // self.ctx.downsample_factor
                video = [f.resize((w, h), Image.LANCZOS) for f in video]
                gt = [f.resize((w, h), Image.LANCZOS) for f in gt]
                output_video = [f.resize((w, h), Image.LANCZOS) for f in output_video]

            logging_dir = f"{self.ctx.logging_dir}"
            os.makedirs(logging_dir, exist_ok=True)

            base_filename = f"T2V"
            output_filename = get_unique_filename(
                base_filename,
                ".mp4",
                prompt=prompt if prompt != '' else 'no_prompt',
                seed=-1,
                resolution=(self.ctx.height, self.ctx.width, self.ctx.num_frames),
                dir=Path(logging_dir),
            )

            if self.ctx.save_grid:
                video_grid = make_video_grid([video, gt, output_video], nrow=3)

                output_filename = output_filename.with_name(
                    f"{output_filename.stem}_{output_filename.suffix}")
                output_filename = str(output_filename.with_suffix(''))
                output_filename = f"{output_filename}"
                # output_filename = output_filename.split("/")[-0]
                output_filename = f"{output_filename}/{video_name}"
                os.makedirs(output_filename, exist_ok=True)

                output_filename = f"{output_filename}/{convert_prompt_to_filename(prompt if prompt != '' else 'no_prompt', max_len=50)}"
                os.makedirs(output_filename, exist_ok=True)
                output_filename_full_video = f"{output_filename}/full_video_{self.ctx.exp_name}_{self.ctx.seed}_{convert_prompt_to_filename(prompt, max_len=50)}_{video_name}.mp4"
                output_filename = f"{output_filename}/{self.ctx.exp_name}_{self.ctx.num_inference_steps}__{self.ctx.guidance_scale}_{self.ctx.seed}_{convert_prompt_to_filename(prompt, max_len=50)}_{video_name}.mp4"
                export_to_video(video_grid, output_filename, fps=self.ctx.frame_rate)

            if self.ctx.individual_videos:
                output_dir_real = f"{logging_dir}/{self.ctx.exp_name}/real"
                output_dir_generated = f"{logging_dir}/{self.ctx.exp_name}/generated"
                os.makedirs(output_dir_real, exist_ok=True)
                os.makedirs(output_dir_generated, exist_ok=True)
                if self.ctx.save_npy:
                    output_filename_real = f"{output_dir_real}/{batch_idx}.npy"
                    output_filename_generated = f"{output_dir_generated}/{batch_idx}.npy"

                    gt_uint8 = np.stack(
                        [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in gt],
                        axis=0
                    )

                    output_video_uint8 = np.stack(
                        [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in output_video],
                        axis=0
                    )

                    np.save(output_filename_real, gt_uint8)
                    np.save(output_filename_generated, output_video_uint8)
                else:
                    os.makedirs(f"{output_dir_real}/{self.ctx.exp_name}", exist_ok=True)
                    os.makedirs(f"{output_dir_generated}/{self.ctx.exp_name}", exist_ok=True)
                    output_filename_real = f"{output_dir_real}/{self.ctx.exp_name}/{batch_idx}.mp4"
                    output_filename_generated = f"{output_dir_generated}/{self.ctx.exp_name}/{batch_idx}.mp4"
                    export_to_video(gt, output_filename_real, fps=self.ctx.frame_rate)
                    export_to_video(output_video, output_filename_generated, fps=self.ctx.frame_rate)

            if self.ctx.save_caption:
                os.makedirs(f"{logging_dir}/{self.ctx.exp_name}/caption", exist_ok=True)
                Path(f"{logging_dir}/{self.ctx.exp_name}/caption/{batch_idx}.txt").write_text(prompt, encoding="utf-8")

            if self.ctx.save_full_video:
                full_video = gt + video
                export_to_video(full_video, output_filename_full_video, fps=self.ctx.frame_rate)


@register("RawDataInferencePipelineFlowC2S")
class RawDataInferencePipelineFlowC2S(RawDataInferencePipeline):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)

    @staticmethod
    def interpolate_frames(video, target_frames: int, mode='linear'):
        B, C, F_, H, W = video.shape

        # Reshape to apply interpolation on frame dimension
        # Merge batch and channels: [B * C, F, H, W]
        video_reshaped = video.view(B * C, F_, H, W)

        # Interpolate over the frame (time) dimension
        # So we unsqueeze to [B*C, 1, F, H, W] and interpolate on dim=2
        video_reshaped = video_reshaped.unsqueeze(1)  # [B*C, 1, F, H, W]

        video_interp = F.interpolate(video_reshaped, size=(target_frames, H, W), mode=mode,
                                     align_corners=False)

        # Remove singleton dim and reshape back to [B, C, target_frames, H, W]
        video_interp = video_interp.squeeze(1).view(B, C, target_frames, H, W)

        return video_interp

    @torch.no_grad()
    def prepare_model_input(self, video_path: Union[str, list[str]]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        video = load_video(video_path) if isinstance(video_path, str) else video_path
        video_tensor = [load_image_to_tensor_with_resize_and_crop(item, self.ctx.height, self.ctx.width) for item in
                        video]
        video_tensor = torch.cat(video_tensor, dim=2)
        video_tensor_input = video_tensor[:, :, :self.ctx.num_frames, :, :]
        _, _, self.num_frames, self.height, self.width = video_tensor_input.shape

        if video_tensor_input.shape[2] != self.ctx.num_frames:
            video_tensor_input = self.interpolate_frames(video_tensor_input,
                                                         self.ctx.num_frames,
                                                         mode='trilinear')

        latents = self.pipe.vae.encode(
            video_tensor_input.to(dtype=self.pipe.vae.dtype, device=self.device)).latent_dist.mode()
        latents = self.pipe._normalize_latents(latents, self.pipe.vae.latents_mean, self.pipe.vae.latents_std)
        model_input = self.pipe._pack_latents(latents,
                                              self.pipe.transformer_spatial_patch_size,
                                              self.pipe.transformer_temporal_patch_size)

        model_inputs = {
            "model_input": model_input
        }

        try:
            future_tensor = video_tensor[:, :, self.ctx.num_frames:2 * self.ctx.num_frames]
            if future_tensor.shape[2] != self.ctx.num_frames:
                future_tensor = torch.zeros_like(video_tensor_input)
        except:
            future_tensor = torch.zeros_like(video_tensor_input)

        return video_tensor_input, future_tensor, model_inputs


@register("RawDataInferencePipelineFlowC2SNuscences")
class RawDataInferencePipelineFlowC2SNuscences(RawDataInferencePipelineFlowC2S):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)

    @staticmethod
    def chunk_list(lst: list, chunk_size: int):
        return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]

    @staticmethod
    def chunk_scene_sweeps(scene_sweeps: dict, chunk_size: int = 82,
                           sample_n: Optional[int] = None):
        chunked = {}
        for scene_token, ts_list in scene_sweeps.items():
            # print("ts_list", len(ts_list))
            chunks = RawDataInferencePipelineFlowC2SNuscences.chunk_list(ts_list, chunk_size)
            for idx, sublist in enumerate(chunks):
                key = f"{scene_token}_chunk_{idx}"
                chunked[key] = sublist

        valid_chunked = {}
        for k, v in chunked.items():
            if len(v) == chunk_size:
                valid_chunked[k] = v

        del chunked

        if sample_n is not None:
            random.seed(4543543)
            keys = list(valid_chunked.keys())
            sampled_keys = random.sample(keys, sample_n)
            valid_chunked = {k: valid_chunked[k] for k in sampled_keys}

        return valid_chunked

    @staticmethod
    def predata_loading(dataroot: str = "./nuscenes",
                        dataset_path: str = "./validation_2k.json",
                        chunk_size: Optional[int] = 82,
                        number_of_samples: Optional[int] = 2000):

        if not os.path.exists(dataset_path):
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils import splits
            nusc = NuScenes(version='v1.0-trainval',
                            dataroot=dataroot,
                            verbose=False)
            val_scene_tokens = [
                {scene['name']: scene['token']} for scene in nusc.scene
                if scene['name'] in splits.val
            ]

            scene_sweeps = {}

            for mapping in val_scene_tokens:
                scene_name, scene_token = next(iter(mapping.items()))
                scene_record = nusc.get('scene', scene_token)

                timestamps = []
                sample_token = scene_record['first_sample_token']

                while sample_token:
                    sample_rec = nusc.get('sample', sample_token)

                    # sd_token = sample_rec['data']["CAM_FRONT"]
                    sd_token = sample_rec['data']["CAM_BACK"]

                    while sd_token:
                        sd_rec = nusc.get('sample_data', sd_token)
                        kt = "Keyframe" if sd_rec['is_key_frame'] else "Sweep"
                        if kt == "Sweep":
                            timestamps.append(sd_rec['filename'])
                        # timestamps.append(sd_rec['filename'])
                        sd_token = sd_rec['next']

                    sample_token = sample_rec['next']

                scene_sweeps[scene_token] = timestamps

            for k, v in scene_sweeps.items():
                scene_sweeps[k] = [f"{dataroot}/{item}" for item in v]
                scene_sweeps[k] = scene_sweeps[k]

            for k, v in scene_sweeps.items():
                scene_sweeps[k] = sorted(list(set(v)))

            if chunk_size is not None:
                chunked = RawDataInferencePipelineFlowC2SNuscences.chunk_scene_sweeps(scene_sweeps,
                                                                                      chunk_size=chunk_size,
                                                                                      sample_n=number_of_samples)
            else:
                chunked = scene_sweeps

            rows = []

            for k, v in chunked.items():
                rows.append({"TOKEN_ID": k, "video": v})

            df = pd.DataFrame(rows)
            df.to_json(dataset_path, orient='records', indent=4)

    def load_data(self, dataset_path, video_column, caption_column) -> None:
        dataset = DatasetNuscencesVal(dataset_path, video_column, caption_column)

        def collate_fn(batch):
            video_paths = []
            prompts = []

            for sample in batch:
                video_paths.append(sample["video_paths"])
                prompts.append(sample["prompts"])

            return {
                "video_paths": video_paths,
                "prompts": prompts,
            }

        self.dataloader = DataLoader(dataset,
                                     batch_size=1,
                                     num_workers=0,
                                     pin_memory=False,
                                     collate_fn=collate_fn)


@register("RawDataInferencePipelineFlowC2SWAN")
class RawDataInferencePipelineFlowC2SWAN(InferencePipeline):
    get_memory_usage = RawDataInferencePipeline.__dict__["get_memory_usage"]
    interpolate_frames = RawDataInferencePipelineFlowC2S.__dict__["interpolate_frames"]

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)

    def load_pipeline(self,
                      pretrained_model_name_or_path: str,
                      transformer_path: str,
                      device: str = "cuda"
                      ) -> None:
        transformer = WanTransformer3DModel.from_pretrained(transformer_path,
                                                            subfolder="transformer",
                                                            torch_dtype=torch.float32)

        self.pipe = WanPipeline.from_pretrained(pretrained_model_name_or_path,
                                                transformer=transformer,
                                                torch_dtype=torch.bfloat16)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    def load_data(self, dataset_path, video_column, caption_column) -> None:
        dataset = DatasetVal(dataset_path, video_column, caption_column)
        self.dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)

    @torch.no_grad()
    def prepare_model_input(self, video_path: Union[str, list[str]]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        video = load_video(video_path) if isinstance(video_path, str) else video_path
        video_tensor = [load_image_to_tensor_with_resize_and_crop(item, self.ctx.height, self.ctx.width) for item in
                        video]
        video_tensor = torch.cat(video_tensor, dim=2)
        video_tensor_input = video_tensor[:, :, :self.ctx.num_frames, :, :]
        _, _, self.num_frames, self.height, self.width = video_tensor_input.shape

        if video_tensor_input.shape[2] != self.ctx.num_frames:
            video_tensor_input = self.interpolate_frames(video_tensor_input,
                                                         self.ctx.num_frames,
                                                         mode='trilinear')

        latents = self.pipe.vae.encode(
            video_tensor_input.to(dtype=self.pipe.vae.dtype, device=self.device)).latent_dist.sample()
        model_input = _normalize_latents_wan(latents,
                                             self.pipe.vae.config.latents_mean,
                                             self.pipe.vae.config.latents_std,
                                             self.pipe.vae.config.z_dim)

        model_inputs = {
            "model_input": model_input
        }

        try:
            future_tensor = video_tensor[:, :, self.ctx.num_frames:2 * self.ctx.num_frames]
            if future_tensor.shape[2] != self.ctx.num_frames:
                future_tensor = torch.zeros_like(video_tensor_input)
        except:
            future_tensor = torch.zeros_like(video_tensor_input)

        return video_tensor_input, future_tensor, model_inputs

    @torch.no_grad()
    def measure_generation_memory(self, batch) -> (float, float, int):
        total_mem = torch.cuda.memory_allocated()  # bytes
        print(f"Total GPU memory used in measurer generation memory: {total_mem / (1024 ** 3):.3f} GB")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.pipe.to(self.device)

        generator = torch.Generator(device=self.device).manual_seed(self.ctx.seed) if self.ctx.seed else None
        prompt = batch["prompts"][0]
        video_path = batch["video_paths"][0]
        video_tensor, gt_tensor, model_inputs = self.prepare_model_input(video_path)

        batch_size = video_tensor.shape[0]
        f = (self.num_frames - 1) // 3 + 1
        h = self.height // 8
        w = self.width // 8
        effective_volume = 2 * 16 * f * h * w

        generation_kwargs = {
            "latents": model_inputs["model_input"],
            "prompt": prompt,
            "negative_prompt": self.ctx.negative_prompt,
            "num_inference_steps": self.ctx.num_inference_steps,
            "num_videos_per_prompt": 1,
            "guidance_scale": self.ctx.guidance_scale,
            "generator": generator,
            "callback_on_step_end": None,
            "height": self.ctx.height,
            "width": self.ctx.width,
            "num_frames": self.ctx.num_frames,
            "output_type": "latent",
        }

        torch.cuda.synchronize()

        # Warm-up
        _ = self.pipe(**generation_kwargs)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

        baseline_mem = torch.cuda.memory_allocated()

        # Generation
        latents = self.pipe(**generation_kwargs).frames

        torch.cuda.synchronize()

        peak_mem = torch.cuda.max_memory_allocated()
        overhead_MB = (peak_mem - baseline_mem) / (1024 ** 2)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        baseline_mem = torch.cuda.memory_allocated()

        decode_standalone_wan(self.pipe, latents, "pil")

        torch.cuda.synchronize()

        peak_mem = torch.cuda.max_memory_allocated()
        overhead_MB_VAE = (peak_mem - baseline_mem) / (1024 ** 2)

        return overhead_MB, overhead_MB_VAE, effective_volume

    @torch.no_grad()
    def infer(self) -> None:
        self.pipe.to(self.device)

        if self.ctx.random_seeds:
            self.ctx.seed = random.randint(0, 1000000000)

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Validation")):
            if batch_idx < self.ctx.starting_idx:
                continue

            if batch_idx >= self.ctx.max_num_of_generated_videos:
                break

            generator = torch.Generator(device=self.device).manual_seed(self.ctx.seed) if self.ctx.seed else None
            prompt = batch["prompts"][0]

            video_path = batch["video_paths"][0]
            if type(video_path) is str:
                video_name = video_path.split("/")[-1].split(".mp4")[0]
            elif type(video_path) is list:
                video_name = video_path[0].split("/")[-1].split(".jpg")[0]
            else:
                video_name = "the_nameless"

            video_tensor, gt_tensor, model_inputs = self.prepare_model_input(video_path)
            output_video = self.pipe(
                latents=model_inputs["model_input"],
                prompt=prompt,
                negative_prompt=self.ctx.negative_prompt,
                num_inference_steps=self.ctx.num_inference_steps,
                num_videos_per_prompt=1,
                guidance_scale=self.ctx.guidance_scale,
                generator=generator,
                callback_on_step_end=None,
                height=self.ctx.height,
                width=self.ctx.width,
                num_frames=self.ctx.num_frames,
                output_type="pil"
            ).frames[0]

            video_tensor_denormalized = (video_tensor * 0.5 + 0.5).clamp(0, 1)
            video_tensor_denormalized = video_tensor_denormalized.permute(0, 2, 1, 3, 4)
            video = [to_pil_image(frame) for frame in video_tensor_denormalized[0]]

            gt_denormalized = (gt_tensor * 0.5 + 0.5).clamp(0, 1)
            gt_tensor_denormalized = gt_denormalized.permute(0, 2, 1, 3, 4)
            gt = [to_pil_image(frame) for frame in gt_tensor_denormalized[0]]

            w, h = output_video[0].size
            if self.ctx.downsample_factor > 1:
                w, h = w // self.ctx.downsample_factor, h // self.ctx.downsample_factor
                video = [f.resize((w, h), Image.LANCZOS) for f in video]
                gt = [f.resize((w, h), Image.LANCZOS) for f in gt]
                output_video = [f.resize((w, h), Image.LANCZOS) for f in output_video]

            video_grid = make_video_grid([video, gt, output_video], nrow=3)

            logging_dir = f"{self.ctx.logging_dir}"
            os.makedirs(logging_dir, exist_ok=True)

            base_filename = f"T2V"
            output_filename = get_unique_filename(
                base_filename,
                ".mp4",
                prompt=prompt if prompt != '' else 'no_prompt',
                seed=-1,
                resolution=(self.ctx.height, self.ctx.width, self.ctx.num_frames),
                dir=Path(logging_dir),
            )

            if self.ctx.save_grid:
                output_filename = output_filename.with_name(
                    f"{output_filename.stem}_{output_filename.suffix}")
                output_filename = str(output_filename.with_suffix(''))
                output_filename = f"{output_filename}"
                output_filename = f"{output_filename}/{video_name}"
                os.makedirs(output_filename, exist_ok=True)

                output_filename = f"{output_filename}/{convert_prompt_to_filename(prompt if prompt != '' else 'no_prompt', max_len=50)}"
                os.makedirs(output_filename, exist_ok=True)
                output_filename_full_video = f"{output_filename}/full_video_{self.ctx.exp_name}_{self.ctx.seed}_{convert_prompt_to_filename(prompt, max_len=50)}_{video_name}.mp4"
                output_filename = f"{output_filename}/{self.ctx.exp_name}_{self.ctx.num_inference_steps}__{self.ctx.guidance_scale}_{self.ctx.seed}_{convert_prompt_to_filename(prompt, max_len=50)}_{video_name}.mp4"
                export_to_video(video_grid, output_filename, fps=self.ctx.frame_rate)

            if self.ctx.individual_videos:
                output_dir_real = f"{logging_dir}/real"
                output_dir_generated = f"{logging_dir}/generated"
                os.makedirs(output_dir_real, exist_ok=True)
                os.makedirs(output_dir_generated, exist_ok=True)

                if self.ctx.save_npy:
                    output_filename_real = f"{output_dir_real}/{batch_idx}.npy"
                    output_filename_generated = f"{output_dir_generated}/{batch_idx}.npy"

                    gt_uint8 = np.stack(
                        [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in gt],
                        axis=0
                    )

                    output_video_uint8 = np.stack(
                        [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in output_video],
                        axis=0
                    )

                    np.save(output_filename_real, gt_uint8)
                    np.save(output_filename_generated, output_video_uint8)
                else:
                    output_filename_real = f"{output_dir_real}/{batch_idx}.mp4"
                    output_filename_generated = f"{output_dir_generated}/{batch_idx}.mp4"
                    export_to_video(gt, output_filename_real, fps=self.ctx.frame_rate)
                    export_to_video(output_video, output_filename_generated, fps=self.ctx.frame_rate)

            if self.ctx.save_full_video:
                full_video = gt + output_video
                export_to_video(full_video, output_filename_full_video, fps=self.ctx.frame_rate)


@register("RawDataInferencePipelineFlowC2SWANNuscences")
class RawDataInferencePipelineFlowC2SWANNuscences(RawDataInferencePipelineFlowC2SWAN):
    chunk_list = RawDataInferencePipelineFlowC2SNuscences.__dict__["chunk_list"]
    chunk_scene_sweeps = RawDataInferencePipelineFlowC2SNuscences.__dict__["chunk_scene_sweeps"]
    predata_loading = RawDataInferencePipelineFlowC2SNuscences.__dict__["predata_loading"]
    load_data = RawDataInferencePipelineFlowC2SNuscences.__dict__["load_data"]

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)
