import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gradio_demo"))

from PIL import Image

if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
from torch import nn
from transformers import AutoTokenizer
import numpy as np
from gradio_demo.utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
preprocess_device = "cpu"
OUTPUT_SIZE_MULTIPLE = 8
MAX_GENERATION_LONG_SIDE = 2048
MAX_PREPROCESS_LONG_SIDE = 2048
USE_BNB_INT8_LINEAR = True
REQUIRE_BNB_INT8_LINEAR = False
BNB_INT8_THRESHOLD = 6.0


def round_to_multiple(value, multiple=OUTPUT_SIZE_MULTIPLE):
    return max(multiple, int(round(value / multiple) * multiple))


def floor_to_multiple(value, multiple=OUTPUT_SIZE_MULTIPLE):
    return max(multiple, int(value // multiple) * multiple)


def get_constrained_size(source_size, max_long_side):
    width, height = source_size
    scale = min(max_long_side / max(width, height), 1.0) if max_long_side else 1.0
    scaled_width = width * scale
    scaled_height = height * scale

    out_width = round_to_multiple(scaled_width)
    out_height = round_to_multiple(scaled_height)

    if max_long_side and max(out_width, out_height) > max_long_side:
        out_width = floor_to_multiple(scaled_width)
        out_height = floor_to_multiple(scaled_height)

    return out_width, out_height


def get_generation_size(source_size):
    return get_constrained_size(source_size, MAX_GENERATION_LONG_SIDE)


def get_preprocess_size(source_size):
    return get_constrained_size(source_size, MAX_PREPROCESS_LONG_SIDE)


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j] == True:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


def get_bnb_linear8bit_class(bnb):
    if hasattr(bnb, "nn") and hasattr(bnb.nn, "Linear8bitLt"):
        return bnb.nn.Linear8bitLt

    import importlib

    for module_name in ("bitsandbytes.nn", "bitsandbytes.nn.modules", "bitsandbytes.modules"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "Linear8bitLt"):
            return module.Linear8bitLt

    package_path = ", ".join(str(path) for path in getattr(bnb, "__path__", [])) or str(getattr(bnb, "__file__", "unknown"))
    raise AttributeError(
        "Could not find bitsandbytes Linear8bitLt. The installed bitsandbytes package appears to be missing "
        f"its Python nn modules. bitsandbytes location: {package_path}"
    )


def make_bnb_linear8bit_compatible(linear8bit_cls):
    class Linear8bitLtCompat(linear8bit_cls):
        def forward(self, input, *args, **kwargs):
            return super().forward(input)

    return Linear8bitLtCompat


def replace_linear_with_bnb_int8(module, module_name, linear8bit_cls):
    converted = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            quantized = linear8bit_cls(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=BNB_INT8_THRESHOLD,
            )
            quantized.load_state_dict(child.state_dict())
            quantized.requires_grad_(False)
            quantized.train(child.training)
            setattr(module, child_name, quantized)
            converted += 1
        else:
            converted += replace_linear_with_bnb_int8(child, f"{module_name}.{child_name}", linear8bit_cls)

    return converted


def apply_bnb_int8_linear_quantization(modules):
    if not USE_BNB_INT8_LINEAR:
        return 0

    try:
        import bitsandbytes as bnb
    except Exception as exc:
        message = (
            "bitsandbytes int8 requested, but bitsandbytes could not be imported. "
            f"Keeping all modules in fp16. Import error: {exc}"
        )
        if REQUIRE_BNB_INT8_LINEAR:
            raise RuntimeError(message) from exc
        print(message)
        return 0

    try:
        linear8bit_cls = make_bnb_linear8bit_compatible(get_bnb_linear8bit_class(bnb))
    except Exception as exc:
        message = f"bitsandbytes imported, but int8 Linear is unavailable. Keeping all modules in fp16. Error: {exc}"
        if REQUIRE_BNB_INT8_LINEAR:
            raise RuntimeError(message) from exc
        print(message)
        return 0

    total = 0
    for module_name, module in modules:
        try:
            count = replace_linear_with_bnb_int8(module, module_name, linear8bit_cls)
        except Exception as exc:
            message = f"Failed to convert {module_name} Linear layers to bitsandbytes int8: {exc}"
            if REQUIRE_BNB_INT8_LINEAR:
                raise RuntimeError(message) from exc
            print(message)
            count = 0
        total += count
        if count:
            print(f"Replaced {count} Linear layers with bitsandbytes int8 in {module_name}.")

    if USE_BNB_INT8_LINEAR:
        if total:
            print(f"Replaced {total} Linear layers with bitsandbytes int8 total.")
        else:
            print("No Linear layers were replaced with bitsandbytes int8.")

    return total


base_path = 'yisol/IDM-VTON'
example_path = 'E:\\projects\\vton\\idmvton\\IDM-VTON\\gradio_demo\\example'

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
                                    )

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(preprocess_device)
openpose_model = OpenPose(preprocess_device)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

apply_bnb_int8_linear_quantization(
    [
        ("text_encoder", text_encoder_one),
        ("text_encoder_2", text_encoder_two),
        ("image_encoder", image_encoder),
    ]
)

tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    unet_encoder=UNet_Encoder,
    torch_dtype=torch.float16,
)

if torch.cuda.is_available():
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()


def start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    human_img_orig = dict["background"].convert("RGB")

    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        out_size = get_generation_size(crop_size)
        final_size = crop_size
        human_img = cropped_img.resize(out_size)
    else:
        out_size = get_generation_size(human_img_orig.size)
        final_size = out_size
        human_img = human_img_orig.resize(out_size)

    out_w, out_h = out_size
    preprocess_size = get_preprocess_size(human_img.size)
    pre_w, pre_h = preprocess_size
    preprocess_img = human_img.resize(preprocess_size)
    garm_img = garm_img.convert("RGB").resize(out_size)

    if is_checked:
        keypoints = openpose_model(preprocess_img, resolution=None)
        model_parse, _ = parsing_model(preprocess_img)
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints, width=pre_w, height=pre_h)
        mask = mask.resize(out_size)
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize(out_size))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(preprocess_img)
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        (
            'show',
            str(PROJECT_ROOT / 'configs' / 'densepose_rcnn_R_50_FPN_s1x.yaml'),
            str(PROJECT_ROOT / 'ckpt' / 'densepose' / 'model_final_162be9.pkl'),
            'dp_segm',
            '-v',
            '--opts',
            'MODEL.DEVICE',
            preprocess_device,
        )
    )
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize(out_size)

    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device, torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device, torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                        cloth=garm_tensor.to(device, torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=out_h,
                        width=out_w,
                        ip_adapter_image=garm_img,
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(final_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray.resize(final_size)
    else:
        return images[0].resize(final_size), mask_gray.resize(final_size)
    # return images[0], mask_gray

human = Image.open("person.jpg").convert("RGB")
garment = Image.open("shirt_trashers.jpg").convert("RGB")

input_dict = {
    "background": human,
    "layers": None,
    "composite": None,
}

result, mask = start_tryon(
    input_dict,
    garment,
    "short sleeve round neck t-shirt",
    True,   # auto mask
    False,  # auto crop
    30,     # denoise steps
    42,     # seed
)

result.save("tryon_result.png")
