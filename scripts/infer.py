import argparse
from inference.pipeline import *
from argparse_dataclass import ArgumentParser
from inference.registry import PIPELINE_REGISTRY

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_pipeline_type", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--device", default="cuda", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)

    known_args, unknown_args = parser.parse_known_args()

    ctx_parser = ArgumentParser(Context)
    ctx = ctx_parser.parse_args(unknown_args)
    data_path = known_args.data_path

    inference_pipe = PIPELINE_REGISTRY.get(known_args.inference_pipeline_type)(known_args.device)

    if known_args.inference_pipeline_type == "RawDataInferencePipelineWAN" or known_args.inference_pipeline_type == "RawDataInferencePipelineWANNuscences":
        ctx.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    inference_pipe.set_ctx(ctx)
    inference_pipe.load_pipeline(known_args.pretrained_model_name_or_path,
                                 known_args.transformer_path)

    if hasattr(inference_pipe, 'predata_loading') and callable(getattr(inference_pipe, 'predata_loading')):
        inference_pipe.predata_loading(dataroot="./nuscenes",
                                       dataset_path=data_path,
                                       chunk_size=82,
                                       number_of_samples=2000)

    inference_pipe.load_data(data_path, "video", "caption")

    inference_pipe.infer()
