# FlowC2S: Flowing from Current to Succeeding Frames for Fast and Memory-Efficient Video Continuation
[//]: # ([![License: MIT]&#40;https://img.shields.io/badge/License-MIT-yellow.svg&#41;]&#40;LICENSE&#41;)

[//]: # ([![Python 3.10+]&#40;https://img.shields.io/badge/python-3.10%2B-blue.svg&#41;]&#40;&#41;)

---

[![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B)](https://arxiv.org/pdf/2604.17625)


## 📄 Abstract

This paper introduces a novel methodology for generating fast and memory-efficient video continuations. Our method, dubbed FlowC2S, fine-tunes a pre-trained text-to-video flow model to learn a vector field between the current and succeeding video chunks. Two design choices are key. First, we introduce inherent optimal couplings, utilizing temporally adjacent video chunks during training as a practical proxy for true optimal couplings, resulting in straighter flows. Second, we incorporate target inversion, injecting the inverted latent of the target chunk into the input representation to strengthen correspondences and improve visual fidelity. By flowing directly from current to succeeding frames, instead of the common combination of current frames with noise to generate a video continuation, we reduce the dimensionality of the model input by a factor of two. The proposed method, fine-tuned from LTXV and Wan, surpasses the state-of-the-art scores across quantitative evaluations with FID and FVD, with as few as five neural function evaluations.

---

## 📊 Results

### Qualitative Examples

![](assets/fig_3_7/4.gif)
![](assets/fig_3_7/81.gif)
![](assets/fig_3_7/44.gif)
![](assets/fig_3_7/932.gif)

### Ablation on Design Choices

[//]: # (![]&#40;assets/plots/combined_loss_FID_FVD.png&#41;)
![](assets/fig_5_8/21.gif)
![](assets/fig_5_8/77.gif)

### Ablation on Neural Function Evaluations (NFE) and Number of Frames

[//]: # (![]&#40;assets/plots/merged_nfe_and_long_video_pred_41.png&#41;)

#### Ablation on NFE

![](assets/fig_9/39.gif)
![](assets/fig_9/86.gif)

#### Ablation on Long Video Continuation

![](assets/fig_10/13.gif)
![](assets/fig_10/74.gif)

### Failure Cases on Long Video Continuation

![](assets/fig_11/9.gif)


## 🛠️ Installation

Clone the repo and create an environment using the requirements.txt
```bash
git clone https://github.com/your-username/flowc2s.git
cd flowc2s
conda env create -f environment.yml -n flowc2s
conda activate flowc2s
```

### ⚡ Inference with LTXV-based model

```bash
export PYTHONPATH=.
python scripts/infer.py \
  --transformer_path "./models/FlowC2S-ltxv" \
  --logging_dir "./video_continuation_results" \
  --exp_name "FlowC2S" \
  --num_inference_steps 10 \
  --cfg 3.5 \
  --num_frames 41 \
  --height 256 \
  --width 384 \
  --downsample_factor 1 \
  --pretrained_model_name_or_path "Lightricks/LTX-Video-0.9.5" \
  --device "cuda" \
  --inference_pipeline_type RawDataInferencePipelineFlowC2S \
  --max_num_of_generated_videos 2001 \
  --data_path "./datasets/openvid-2k-validation.json" \
  --starting_idx 0 \
  --individual_videos \
  --save_grid \
  --individual_videos
```

## ⚙️ Training 

Use [LTXV Trainer](https://github.com/Lightricks/LTX-Video-Trainer) for dataset preparation and precomputation. 

### Fine-tuning from LTXV

```bash
export PYTHONPATH=.
accelerate --mixed_precision bf16 scripts/.py \
  --pretrained_model_name_or_path "Lightricks/LTX-Video-0.9.5" \
  --text_encoder_model_name_or_path "PixArt-alpha/PixArt-XL-2-1024-MS" \
  --logging_dir "logs" \
  --video_init_dataset_root "path/to/precomputed/initial/data/distribution" \
  --video_data_dataset_root "path/to/precomputed/data/data/distribution" \
  --validation_init_video_dataset_root "path/to/validation/precomputed/initial/data/distribution" \
  --validation_data_video_dataset_root "path/to/validation/precomputed/data/data/distribution" \
  --train_dataset_path "" \
  --validation_dataset_path "path/to/validation/val.json" \
  --video_reshape_mode  "center" \
  --output_dir "./experiments/FlowC2S" \
  --caption_column "caption" \
  --video_column "video" \
  --tracker_name "FlowC2S" \
  --seed 779878798 \
  --seed_x1 4324421 \
  --mixed_precision bf16 \
  --transformer_dtype f32 \
  --validation_height 256 --validation_width 384 --fps 25 --validation_num_frames 45 --skip_frames_start 0 --skip_frames_end 0 \
  --max_num_frames 145 \
  --height_buckets 256 \
  --width_buckets 384 \
  --frame_buckets 45 \
  --frame_rate 25 \
  --train_batch_size 64 \
  --validation_batch_size 1 \
  --max_train_steps 1450 \
  --checkpointing_steps 38 \
  --validation_steps 38 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing \
  --learning_rate 2e-4 \
  --lr_scheduler linear \
  --lr_warmup_steps 30 \
  --lr_num_cycles 1 \
  --optimizer adamw \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --max_grad_norm 1.0 \
  --validation_negative_prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --validation_num_inference_steps 50 \
  --validation_guidance_scale 3.5 \
  --validation_strength 2.0 \
  --report_to wandb \
  --dataset_type "precomputed" \
  --sigma_sampler_type "ShiftedLogitNormalTimestepSampler" \
  --dataloader_num_workers 20 \
  --validation_only_caption \
  --dist_regularization_prob 0.7 \
  --offload
```

## Citations

If FlowC2S contributes to your research, please cite the paper:

```
@article{margaryan2026flowc2s,
  title   = {FlowC2S: Flowing from Current to Succeeding Frames for Fast and Memory-Efficient Video Continuation},
  author  = {Hovhannes Margaryan and Quentin Bammey and Christian Sandor},
  journal = {arXiv preprint arXiv:2604.17625},
  year    = {2026},
}
```

## Acknowledgments

We thank the authors of [Diffusers](https://github.com/huggingface/diffusers) and [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) for their valuable open-source contributions.
We also acknowledge the broader open-source ecosystem (e.g., PyTorch, Hugging Face, etc.) that made our research possible.

## Contact

This repository is under development. If you encounter any issues or have questions, please open a GitHub issue or reach out via email at [marg.hovo@gmail.com](mailto:marg.hovo@gmail.com).