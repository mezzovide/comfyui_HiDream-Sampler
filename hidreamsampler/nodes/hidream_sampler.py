from ..config import (
    MODEL_CONFIGS,
    RESOLUTION_OPTIONS,
)
from ..helpers import (
    load_models,
    parse_resolution,
    pil2tensor,
)
import torch
from PIL import Image
import comfy.utils
import logging

from .hidream_base import HiDreamBase

# Set up logger for this module
logger = logging.getLogger(__name__)


# --- ComfyUI Node Definition ---
class HiDreamSampler(HiDreamBase):
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"

    cleanup_models = HiDreamBase.cleanup_models

    @staticmethod
    def parse_aspect_ratio(aspect_ratio_str):
        """Parse aspect ratio string to get width and height"""
        try:
            # Extract dimensions from the parenthesis
            dims_part = aspect_ratio_str.split("(")[1].split(")")[0]
            width, height = dims_part.split("×")
            return int(width), int(height)
        except Exception as e:
            logger.error(
                f"Error parsing aspect ratio '{aspect_ratio_str}': {e}. Falling back to 1024x1024."
            )
            return 1024, 1024

    @classmethod
    def INPUT_TYPES(s):
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types:
            return {
                "required": {
                    "error": (
                        "STRING",
                        {"default": "No models available...", "multiline": True},
                    )
                }
            }
        default_model = (
            "fast-nf4"
            if "fast-nf4" in available_model_types
            else "fast"
            if "fast" in available_model_types
            else available_model_types[0]
        )

        # Define schedulers
        scheduler_options = [
            "Default for model",
            "UniPC",
            "Euler",
            "Karras Euler",
            "Karras Exponential",
        ]

        return {
            "required": {
                "model_type": (available_model_types, {"default": default_model}),
                "prompt": ("STRING", {"multiline": True, "default": "..."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "1024 × 1024 (Square)"}),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 8,
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "scheduler": (scheduler_options, {"default": "Default for model"}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
                "override_cfg": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1},
                ),
                "override_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 8},
                ),
                "override_height": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 8},
                ),
            }
        }

    def generate(
        self,
        model_type,
        prompt,
        negative_prompt,
        resolution,
        num_images,
        seed,
        scheduler,
        override_steps,
        override_cfg,
        override_width,
        override_height,
    ):
        use_uncensored_llm = None

        # Determine resolution
        if override_width > 0 and override_height > 0:
            height, width = override_height, override_width
            logger.info(f"Using override resolution: {width}x{height}")
        else:
            height, width = parse_resolution(resolution)
            logger.info(f"Using fixed resolution: {width}x{height} ({resolution})")

        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"HiDream: Initial VRAM usage: {initial_mem:.2f} MB")
        if not MODEL_CONFIGS or model_type == "error":
            logger.error("HiDream Error: No models loaded.")
            return (torch.zeros((1, 512, 512, 3)),)

        cache_key = f"{model_type}"
        pipe, config = self.get_model_from_cache(cache_key, self._model_cache)

        if pipe is None:
            self.clear_model_cache(self._model_cache)
            logger.info(f"Loading model for {model_type}...")
            try:
                pipe, config = load_models(model_type, use_uncensored_llm)
                self._model_cache[cache_key] = (pipe, config)
                logger.info(
                    f"Model {model_type}{' (uncensored)' if use_uncensored_llm else ''} loaded & cached!"
                )
            except Exception as e:
                logger.error(f"!!! ERROR loading {model_type}: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return (torch.zeros((1, 512, 512, 3)),)

        if pipe is None or config is None:
            logger.critical("CRITICAL ERROR: Load failed.")
            return (torch.zeros((1, 512, 512, 3)),)

        original_scheduler_class = config["scheduler_class"]
        original_shift = config["shift"]
        self.update_scheduler(
            pipe, config, scheduler, original_scheduler_class, original_shift
        )

        is_nf4_current = config.get("is_nf4", False)
        num_inference_steps = (
            override_steps if override_steps >= 0 else config["num_inference_steps"]
        )
        guidance_scale = (
            override_cfg if override_cfg >= 0.0 else config["guidance_scale"]
        )

        pbar = comfy.utils.ProgressBar(num_inference_steps)

        max_length_clip_l = 77
        max_length_openclip = 150
        max_length_t5 = 256
        max_length_llama = 256

        clip_l_weight = 1.0
        openclip_weight = 1.0
        t5_weight = 1.0
        llama_weight = 1.0

        try:
            inference_device = comfy.model_management.get_torch_device()
        except Exception:
            inference_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        logger.info(f"Creating Generator on: {inference_device}")
        generator = torch.Generator(device=inference_device).manual_seed(seed)
        logger.info("\n--- Starting Generation ---")
        logger.info(
            f"Model: {model_type}, Res: {height}x{width}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}"
        )
        logger.info(
            f"Using standard sequence lengths: CLIP-L: {max_length_clip_l}, OpenCLIP: {max_length_openclip}, T5: {max_length_t5}, Llama: {max_length_llama}"
        )

        pipeline_output = None
        try:
            if not is_nf4_current:
                logger.info(
                    f"Ensuring pipe on: {inference_device} (Offload NOT enabled)"
                )
                pipe.to(inference_device)
            else:
                logger.info(
                    f"Skipping pipe.to({inference_device}) (CPU offload enabled)."
                )
            logger.info("Executing pipeline inference...")

            if num_images > 1:
                logger.info(
                    f"Preparing for batch generation with {num_images} images..."
                )
                output_images_list = []
                for i in range(num_images):
                    logger.info(f"Generating image {i + 1}/{num_images}...")
                    with torch.inference_mode():
                        single_output = pipe(
                            prompt=prompt,
                            prompt_2=prompt,
                            prompt_3=prompt,
                            prompt_4=prompt,
                            negative_prompt=negative_prompt.strip()
                            if negative_prompt
                            else None,
                            height=height,
                            width=width,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            num_images_per_prompt=1,
                            generator=torch.Generator(
                                device=inference_device
                            ).manual_seed(seed + i),
                            max_sequence_length_clip_l=max_length_clip_l,
                            max_sequence_length_openclip=max_length_openclip,
                            max_sequence_length_t5=max_length_t5,
                            max_sequence_length_llama=max_length_llama,
                            clip_l_scale=clip_l_weight,
                            openclip_scale=openclip_weight,
                            t5_scale=t5_weight,
                            llama_scale=llama_weight,
                        )
                    output_images_list.extend(single_output.images)
                    pbar.update_absolute((i + 1) * num_inference_steps // num_images)
            else:
                with torch.inference_mode():
                    pipeline_output = pipe(
                        prompt=prompt,
                        prompt_2=prompt,
                        prompt_3=prompt,
                        prompt_4=prompt,
                        negative_prompt=negative_prompt.strip()
                        if negative_prompt
                        else None,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=num_images,
                        generator=generator,
                        max_sequence_length_clip_l=max_length_clip_l,
                        max_sequence_length_openclip=max_length_openclip,
                        max_sequence_length_t5=max_length_t5,
                        max_sequence_length_llama=max_length_llama,
                        clip_l_scale=clip_l_weight,
                        openclip_scale=openclip_weight,
                        t5_scale=t5_weight,
                        llama_scale=llama_weight,
                    )
                    output_images_list = pipeline_output.images

            logger.info("Pipeline inference finished.")

        except Exception as e:
            logger.error(f"!!! ERROR during execution: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return (torch.zeros((1, height, width, 3)),)
        finally:
            pbar.update_absolute(num_inference_steps)
        logger.info("--- Generation Complete ---")

        # Use base class output processing for the first image (batching can be improved in future)
        if (
            output_images_list is None
            or not isinstance(output_images_list, list)
            or len(output_images_list) == 0
        ):
            logger.error(
                f"ERROR: No images returned or invalid format (Type: {type(output_images_list)}). Creating blank image."
            )
            return (torch.zeros((1, height, width, 3)),)

        # For batch, stack tensors
        try:
            logger.info(f"Processing {len(output_images_list)} output image(s).")
            tensor_list = []
            for i, img in enumerate(output_images_list):
                if not isinstance(img, Image.Image):
                    logger.warning(
                        f"WARNING: Item {i} in output list is not a PIL Image (Type: {type(img)}). Skipping."
                    )
                    continue
                logger.info(f"Converting image {i + 1}/{len(output_images_list)}...")
                single_tensor = pil2tensor(img)
                if single_tensor is not None:
                    if len(single_tensor.shape) == 4 and single_tensor.shape[0] == 1:
                        tensor_list.append(single_tensor)
                    else:
                        logger.warning(
                            f"WARNING: pil2tensor returned unexpected shape {single_tensor.shape} for image {i}. Skipping."
                        )
                else:
                    logger.warning(
                        f"WARNING: pil2tensor failed for image {i}. Skipping."
                    )
            if not tensor_list:
                logger.error(
                    "ERROR: All image conversions failed. Creating blank image."
                )
                return (torch.zeros((1, height, width, 3)),)
            output_tensor = torch.cat(tensor_list, dim=0)
            logger.info(
                f"Successfully converted {output_tensor.shape[0]} images into batch tensor."
            )
            if output_tensor.dtype != torch.float32:
                logger.info(
                    f"Converting batched {output_tensor.dtype} tensor to float32 for ComfyUI compatibility"
                )
                output_tensor = output_tensor.to(torch.float32)
            if (
                len(output_tensor.shape) != 4
                or output_tensor.shape[0] == 0
                or output_tensor.shape[3] != 3
            ):
                logger.error(
                    f"ERROR: Invalid final batch tensor shape {output_tensor.shape}. Creating blank image."
                )
                return (torch.zeros((1, height, width, 3)),)
            logger.info(f"Output tensor shape: {output_tensor.shape}")
            try:
                import comfy.model_management as model_management

                logger.info("HiDream: Requesting ComfyUI memory cleanup...")
                model_management.soft_empty_cache()
            except Exception as e:
                logger.warning(f"HiDream: ComfyUI cleanup failed: {e}")
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated() / 1024**2
                logger.info(
                    f"HiDream: Final VRAM usage: {final_mem:.2f} MB (Change: {final_mem - initial_mem:.2f} MB)"
                )
            return (output_tensor,)
        except Exception as e:
            logger.error(f"Error processing output image: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return (torch.zeros((1, height, width, 3)),)
