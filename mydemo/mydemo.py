import json
import re
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


# Single-image inputs and outputs.
HUMAN_IMAGE_PATH = PROJECT_ROOT / "mydemo" / "person.jpg"
GARMENT_IMAGE_PATH = PROJECT_ROOT / "mydemo" / "shirt_trashers.jpg"
OUTPUT_IMAGE_PATH = PROJECT_ROOT / "mydemo" / "tryon_result.png"
MASK_PREVIEW_PATH = PROJECT_ROOT / "mydemo" / "mask_preview.png"

# Batch inputs and outputs.
HUMAN_DIR = PROJECT_ROOT / "mydemo" / "human"
GARMENT_DIR = PROJECT_ROOT / "mydemo" / "garment"
RESULT_DIR = PROJECT_ROOT / "mydemo" / "result"
GARMENT_LABELS_PATH = PROJECT_ROOT / "mydemo" / "garment_labels.json"
SAVE_BATCH_MASK_PREVIEWS = False
MASK_RESULT_DIR = RESULT_DIR / "mask_preview"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Main switches.
USE_AUTO_GENERATED_MASK = True
ENABLE_AUTO_RESIZE_AND_CROP = True

# Positive values grow the white inpaint/garment mask. Negative values shrink
# it. This is applied after IDM-VTON's original parser/openpose mask and is
# measured at the 768x1024 working size.
AUTO_MASK_PADDING_PIXELS = 0

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


if __name__ == "__main__":
    for image_dir in (HUMAN_DIR, GARMENT_DIR):
        if not image_dir.exists():
            raise SystemExit(f"Missing input folder: {image_dir}")
        if not any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in image_dir.iterdir()):
            raise SystemExit(f"No supported image files found in: {image_dir}")


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
    padding = int(AUTO_MASK_PADDING_PIXELS)
    if padding == 0:
        return mask

    mask_array = np.array(mask.convert("L"))
    mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
    kernel_size = 2 * abs(padding) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if padding > 0:
        mask_array = cv2.dilate(mask_array, kernel, iterations=1)
    else:
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


def start_tryon(
    input_data,
    garm_img,
    garment_des,
    use_auto_mask,
    use_auto_crop,
    denoise_steps,
    seed,
    garment_category=GARMENT_CATEGORY,
):
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
        mask, _ = get_mask_location(MODEL_TYPE, garment_category, model_parse, keypoints)
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


def list_image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected image directory: {directory}")

    image_paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No images found in {directory}")
    return image_paths


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return stem or "image"


def load_garment_labels(labels_path: Path = GARMENT_LABELS_PATH) -> dict[str, dict[str, str]]:
    if not labels_path.exists():
        print(f"Garment label file not found, using fallback description: {labels_path}")
        return {}

    with labels_path.open("r", encoding="utf-8") as labels_file:
        labels = json.load(labels_file)
    if not isinstance(labels, dict):
        raise ValueError(f"Garment label file must contain a JSON object: {labels_path}")
    return labels


def get_garment_label(
    garment_path: Path,
    garment_labels: dict[str, dict[str, str]],
    fallback_description: str,
    fallback_category: str = GARMENT_CATEGORY,
) -> tuple[str, str]:
    metadata = garment_labels.get(garment_path.name, {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Label metadata for {garment_path.name} must be an object.")

    description = metadata.get("description", fallback_description)
    category = metadata.get("category", fallback_category)
    if category not in {"upper_body", "lower_body", "dresses"}:
        raise ValueError(
            f"Unsupported category for {garment_path.name}: {category}. "
            "Use upper_body, lower_body, or dresses."
        )
    return description, category


def run_tryon(
    human_image_path: Path,
    garment_image_path: Path,
    garment_description: str,
    manual_mask_path: Optional[Path] = None,
    garment_category: str = GARMENT_CATEGORY,
    output_image_path: Optional[Path] = OUTPUT_IMAGE_PATH,
    mask_preview_path: Optional[Path] = MASK_PREVIEW_PATH,
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
        garment_category,
    )
    if output_image_path is not None:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_image_path)
    if mask_preview_path is not None:
        mask_preview_path.parent.mkdir(parents=True, exist_ok=True)
        mask_preview.save(mask_preview_path)
    return result, mask_preview


def run_batch_tryon(
    human_dir: Path = HUMAN_DIR,
    garment_dir: Path = GARMENT_DIR,
    result_dir: Path = RESULT_DIR,
    garment_description: str = GARMENT_DESCRIPTION,
):
    human_paths = list_image_paths(human_dir)
    garment_paths = list_image_paths(garment_dir)
    garment_labels = load_garment_labels()
    result_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_BATCH_MASK_PREVIEWS:
        MASK_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(human_paths) * len(garment_paths)
    completed = 0
    for human_path in human_paths:
        for garment_path in garment_paths:
            completed += 1
            pair_name = f"{safe_stem(human_path)}__{safe_stem(garment_path)}.png"
            output_path = result_dir / pair_name
            mask_path = MASK_RESULT_DIR / pair_name if SAVE_BATCH_MASK_PREVIEWS else None
            description, category = get_garment_label(
                garment_path,
                garment_labels,
                garment_description,
            )
            print(
                f"[{completed}/{total}] {human_path.name} + {garment_path.name} "
                f"({description}, {category}) -> {output_path.name}"
            )
            run_tryon(
                human_path,
                garment_path,
                description,
                garment_category=category,
                output_image_path=output_path,
                mask_preview_path=mask_path,
            )


if __name__ == "__main__":
    run_batch_tryon()
