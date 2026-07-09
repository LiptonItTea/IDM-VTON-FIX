import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRADIO_DEMO_ROOT = PROJECT_ROOT / "gradio_demo"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GRADIO_DEMO_ROOT))

if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

import apply_net
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from utils_mask import get_mask_location


# Inputs and outputs.
HUMAN_IMAGE_PATH = PROJECT_ROOT / "mydemo" / "person.jpg"
GARMENT_IMAGE_PATH = PROJECT_ROOT / "mydemo" / "shirt_trashers.jpg"
OUTPUT_IMAGE_PATH = PROJECT_ROOT / "mydemo" / "tryon_result.png"
MASK_PREVIEW_PATH = PROJECT_ROOT / "mydemo" / "mask_preview.png"

# Main switches.
USE_AUTO_GENERATED_MASK = True
ENABLE_AUTO_RESIZE_AND_CROP = False

# Positive values grow the white inpaint/garment mask, negative values are not
# used so the direction stays explicit. These are applied after IDM-VTON's
# original parser/openpose mask and are measured at the 768x1024 working size.
AUTO_MASK_GARMENT_EXPAND_PIXELS = 0
AUTO_MASK_GARMENT_CONTRACT_PIXELS = 0

# Original Gradio demo settings.
BASE_MODEL_PATH = "yisol/IDM-VTON"
GARMENT_DESCRIPTION = "short sleeve round neck t-shirt"
DENOISE_STEPS = 30
SEED = 42
MODEL_TYPE = "hd"
GARMENT_CATEGORY = "upper_body"
MODEL_WIDTH = 768
MODEL_HEIGHT = 1024
PREPROCESS_WIDTH = 384
PREPROCESS_HEIGHT = 512
GUIDANCE_SCALE = 2.0
STRENGTH = 1.0
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
preprocess_device = 0 if torch.cuda.is_available() else "cpu"
densepose_device = "cuda" if torch.cuda.is_available() else "cpu"


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


def apply_auto_mask_thickness(mask: Image.Image) -> Image.Image:
    if AUTO_MASK_GARMENT_EXPAND_PIXELS <= 0 and AUTO_MASK_GARMENT_CONTRACT_PIXELS <= 0:
        return mask

    mask_array = np.array(mask.convert("L"))
    mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)

    if AUTO_MASK_GARMENT_EXPAND_PIXELS > 0:
        expand = int(AUTO_MASK_GARMENT_EXPAND_PIXELS)
        kernel_size = 2 * expand + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_array = cv2.dilate(mask_array, kernel, iterations=1)

    if AUTO_MASK_GARMENT_CONTRACT_PIXELS > 0:
        contract = int(AUTO_MASK_GARMENT_CONTRACT_PIXELS)
        kernel_size = 2 * contract + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_array = cv2.erode(mask_array, kernel, iterations=1)

    return Image.fromarray(mask_array)


unet = UNet2DConditionModel.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_PATH, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="vae",
    torch_dtype=torch.float16,
)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    BASE_MODEL_PATH,
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

tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

pipe = TryonPipeline.from_pretrained(
    BASE_MODEL_PATH,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder


def start_tryon(input_data, garm_img, garment_des, use_auto_mask, use_auto_crop, denoise_steps, seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((MODEL_WIDTH, MODEL_HEIGHT))
    human_img_orig = input_data["background"].convert("RGB")

    if use_auto_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((MODEL_WIDTH, MODEL_HEIGHT))
    else:
        human_img = human_img_orig.resize((MODEL_WIDTH, MODEL_HEIGHT))

    if use_auto_mask:
        keypoints = openpose_model(human_img.resize((PREPROCESS_WIDTH, PREPROCESS_HEIGHT)))
        model_parse, _ = parsing_model(human_img.resize((PREPROCESS_WIDTH, PREPROCESS_HEIGHT)))
        mask, _ = get_mask_location(MODEL_TYPE, GARMENT_CATEGORY, model_parse, keypoints)
        mask = mask.resize((MODEL_WIDTH, MODEL_HEIGHT))
        mask = apply_auto_mask_thickness(mask)
    else:
        layers = input_data.get("layers") or [None]
        manual_mask = layers[0]
        if manual_mask is None:
            raise ValueError("Manual mask mode requires input_data['layers'][0].")
        mask = pil_to_binary_mask(manual_mask.convert("RGB").resize((MODEL_WIDTH, MODEL_HEIGHT)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((PREPROCESS_WIDTH, PREPROCESS_HEIGHT)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        (
            "show",
            str(PROJECT_ROOT / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"),
            str(PROJECT_ROOT / "ckpt" / "densepose" / "model_final_162be9.pkl"),
            "dp_segm",
            "-v",
            "--opts",
            "MODEL.DEVICE",
            densepose_device,
        )
    )
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((MODEL_WIDTH, MODEL_HEIGHT))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
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
                        negative_prompt=NEGATIVE_PROMPT,
                    )

                prompt = "a photo of " + garment_des
                negative_prompt = NEGATIVE_PROMPT
                if not isinstance(prompt, list):
                    prompt = [prompt] * 1
                if not isinstance(negative_prompt, list):
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
                generator = torch.Generator(device).manual_seed(int(seed)) if seed is not None else None
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=STRENGTH,
                    pose_img=pose_img.to(device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor.to(device, torch.float16),
                    mask_image=mask,
                    image=human_img,
                    height=MODEL_HEIGHT,
                    width=MODEL_WIDTH,
                    ip_adapter_image=garm_img.resize((MODEL_WIDTH, MODEL_HEIGHT)),
                    guidance_scale=GUIDANCE_SCALE,
                )[0]

    if use_auto_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray

    return images[0], mask_gray


def run_tryon(
    human_image_path: Path,
    garment_image_path: Path,
    garment_description: str,
    manual_mask_path: Optional[Path] = None,
):
    human = Image.open(human_image_path).convert("RGB")
    garment = Image.open(garment_image_path).convert("RGB")
    manual_mask = Image.open(manual_mask_path).convert("RGB") if manual_mask_path else None

    input_data = {
        "background": human,
        "layers": [manual_mask] if manual_mask is not None else [None],
        "composite": None,
    }

    result, mask_preview = start_tryon(
        input_data,
        garment,
        garment_description,
        USE_AUTO_GENERATED_MASK,
        ENABLE_AUTO_RESIZE_AND_CROP,
        DENOISE_STEPS,
        SEED,
    )
    result.save(OUTPUT_IMAGE_PATH)
    mask_preview.save(MASK_PREVIEW_PATH)
    return result, mask_preview


if __name__ == "__main__":
    run_tryon(
        HUMAN_IMAGE_PATH,
        GARMENT_IMAGE_PATH,
        GARMENT_DESCRIPTION,
    )
