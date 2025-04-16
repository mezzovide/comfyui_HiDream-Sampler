from ..config import (
    MODEL_CONFIGS,
    RESOLUTION_OPTIONS,
)
from ..helpers import (
    global_cleanup,
    get_scheduler_instance,
    load_models,
    parse_resolution,
    pil2tensor,
)
import torch
from PIL import Image
import gc
import comfy.utils

from .hidream_sampler import HiDreamSampler


class HiDreamImg2Img:
    _model_cache = HiDreamSampler._model_cache
    cleanup_models = HiDreamSampler.cleanup_models

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"

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
                "image": ("IMAGE",),
                "denoising_strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "prompt": ("STRING", {"multiline": True, "default": "..."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "scheduler": (scheduler_options, {"default": "Default for model"}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
                "override_cfg": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1},
                ),
                "use_uncensored_llm": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "llm_system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a creative AI assistant that helps create detailed, vivid images based on user descriptions.",
                    },
                ),
                "clip_l_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
                "openclip_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
                "t5_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
                "llama_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
            },
        }

    def preprocess_image(self, image, target_height=None, target_width=None):
        """Resize and possibly crop input image to match model requirements."""
        import torch.nn.functional as F

        # Get original dimensions
        _, orig_h, orig_w, _ = image.shape
        orig_aspect = orig_w / orig_h

        print(
            f"Original image dimensions: {orig_w}x{orig_h}, aspect ratio: {orig_aspect:.3f}"
        )

        # If no target size provided, find closest standard resolution
        if target_height is None or target_width is None:
            # Define standard resolutions (must be divisible by 16)
            standard_resolutions = [
                (1024, 1024),  # 1:1
                (768, 1360),  # 9:16 (portrait)
                (1360, 768),  # 16:9 (landscape)
                (880, 1168),  # 3:4 (portrait)
                (1168, 880),  # 4:3 (landscape)
                (832, 1248),  # 2:3 (portrait)
                (1248, 832),  # 3:2 (landscape)
            ]

            # Find closest aspect ratio
            best_diff = float("inf")
            target_width, target_height = standard_resolutions[0]  # Default to square

            for w, h in standard_resolutions:
                res_aspect = w / h
                diff = abs(res_aspect - orig_aspect)
                if diff < best_diff:
                    best_diff = diff
                    target_width, target_height = w, h

            print(f"Selected target resolution: {target_width}x{target_height}")

        # Ensure dimensions are divisible by 16
        target_width = (target_width // 16) * 16
        target_height = (target_height // 16) * 16

        # Convert to format expected by F.interpolate [B,C,H,W]
        # ComfyUI typically uses [B,H,W,C]
        x = image.permute(0, 3, 1, 2)

        # Calculate resize dimensions preserving aspect ratio
        if orig_aspect > target_width / target_height:  # Image is wider
            new_w = target_width
            new_h = int(new_w / orig_aspect)
            new_h = (new_h // 16) * 16  # Make divisible by 16
        else:  # Image is taller
            new_h = target_height
            new_w = int(new_h * orig_aspect)
            new_w = (new_w // 16) * 16  # Make divisible by 16

        # Resize to preserve aspect ratio
        x_resized = F.interpolate(
            x, size=(new_h, new_w), mode="bicubic", align_corners=False
        )

        # Create target tensor with correct dimensions
        x_result = torch.zeros(
            1, 3, target_height, target_width, device=x.device, dtype=x.dtype
        )

        # Calculate position for center crop
        y_offset = max(0, (new_h - target_height) // 2)
        x_offset = max(0, (new_w - target_width) // 2)

        # Calculate how much to copy
        height_to_copy = min(new_h, target_height)
        width_to_copy = min(new_w, target_width)

        # Place the resized image in the center of the target tensor
        target_y_offset = max(0, (target_height - height_to_copy) // 2)
        target_x_offset = max(0, (target_width - width_to_copy) // 2)

        x_result[
            :,
            :,
            target_y_offset : target_y_offset + height_to_copy,
            target_x_offset : target_x_offset + width_to_copy,
        ] = x_resized[
            :,
            :,
            y_offset : y_offset + height_to_copy,
            x_offset : x_offset + width_to_copy,
        ]

        print(f"Processed to: {target_width}x{target_height} (divisible by 16)")

        # Convert back to ComfyUI format [B,H,W,C]
        return x_result.permute(0, 2, 3, 1)

    def generate(
        self,
        model_type,
        image,
        denoising_strength,
        prompt,
        negative_prompt,
        seed,
        scheduler,
        override_steps,
        override_cfg,
        use_uncensored_llm=False,
        llm_system_prompt="You are a creative AI assistant...",
        clip_l_weight=1.0,
        openclip_weight=1.0,
        t5_weight=1.0,
        llama_weight=1.0,
        **kwargs,
    ):
        # Preprocess the input image to ensure compatible dimensions
        processed_image = self.preprocess_image(image)

        # Get dimensions from processed image for the output
        _, height, width, _ = processed_image.shape

        # Monitor initial memory usage
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1024**2
            print(f"HiDream: Initial VRAM usage: {initial_mem:.2f} MB")

        if not MODEL_CONFIGS or model_type == "error":
            print("HiDream Error: No models loaded.")
            return (torch.zeros((1, 512, 512, 3)),)

        pipe = None
        config = None

        # Create cache key that includes uncensored state
        cache_key = (
            f"{model_type}_img2img_{'uncensored' if use_uncensored_llm else 'standard'}"
        )

        # Try to reuse from cache first
        if cache_key in self._model_cache:
            print(f"Checking cache for {cache_key}...")
            pipe, config = self._model_cache[cache_key]
            valid_cache = True
            if (
                pipe is None
                or config is None
                or not hasattr(pipe, "transformer")
                or pipe.transformer is None
            ):
                valid_cache = False
                print("Invalid cache, reloading...")
                del self._model_cache[cache_key]
                pipe, config = None, None
            if valid_cache:
                print("Using cached model.")

        if pipe is None:
            if self._model_cache:
                print(f"Clearing img2img cache before loading {model_type}...")
                keys_to_del = list(self._model_cache.keys())
                for key in keys_to_del:
                    print(f"  Removing '{key}'...")
                    try:
                        pipe_to_del, _ = self._model_cache.pop(key)
                        # More aggressive cleanup
                        if hasattr(pipe_to_del, "transformer"):
                            pipe_to_del.transformer = None
                        if hasattr(pipe_to_del, "text_encoder_4"):
                            pipe_to_del.text_encoder_4 = None
                        if hasattr(pipe_to_del, "tokenizer_4"):
                            pipe_to_del.tokenizer_4 = None
                        if hasattr(pipe_to_del, "scheduler"):
                            pipe_to_del.scheduler = None
                        del pipe_to_del
                    except Exception as e:
                        print(f"  Error removing {key}: {e}")

                # Multiple garbage collection passes
                for _ in range(3):
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Force synchronization
                    torch.cuda.synchronize()
                print("Cache cleared.")

            print(f"Loading model for {model_type} img2img...")

            try:
                # First load regular model
                txt2img_pipe, config = load_models(model_type, use_uncensored_llm)

                # Convert to img2img pipeline
                print("Creating img2img pipeline from loaded txt2img pipeline...")
                from hi_diffusers.pipelines.hidream_image.pipeline_hidream_image_to_image import (
                    HiDreamImageToImagePipeline,
                )

                pipe = HiDreamImageToImagePipeline(
                    scheduler=txt2img_pipe.scheduler,
                    vae=txt2img_pipe.vae,
                    text_encoder=txt2img_pipe.text_encoder,
                    tokenizer=txt2img_pipe.tokenizer,
                    text_encoder_2=txt2img_pipe.text_encoder_2,
                    tokenizer_2=txt2img_pipe.tokenizer_2,
                    text_encoder_3=txt2img_pipe.text_encoder_3,
                    tokenizer_3=txt2img_pipe.tokenizer_3,
                    text_encoder_4=txt2img_pipe.text_encoder_4,
                    tokenizer_4=txt2img_pipe.tokenizer_4,
                )

                # Copy transformer and move to right device
                pipe.transformer = txt2img_pipe.transformer

                # Cleanup txt2img pipeline references
                txt2img_pipe = None

                # Cache the img2img pipeline
                self._model_cache[cache_key] = (pipe, config)
                print(f"Model {model_type} loaded & cached for img2img!")

            except Exception as e:
                print(f"!!! ERROR loading {model_type}: {e}")
                import traceback

                traceback.print_exc()
                return (torch.zeros((1, 512, 512, 3)),)

        if pipe is None or config is None:
            print("CRITICAL ERROR: Load failed.")
            return (torch.zeros((1, 512, 512, 3)),)

        # Update scheduler if requested
        original_scheduler_class = config["scheduler_class"]
        original_shift = config["shift"]
        if scheduler != "Default for model":
            print(
                f"Replacing default scheduler ({original_scheduler_class}) with: {scheduler}"
            )
            # Create a completely fresh scheduler instance to avoid any parameter leakage
            if scheduler == "UniPC":
                new_scheduler = get_scheduler_instance(
                    "FlowUniPCMultistepScheduler", original_shift
                )
                pipe.scheduler = new_scheduler
            elif scheduler == "Euler":
                new_scheduler = get_scheduler_instance(
                    "FlashFlowMatchEulerDiscreteScheduler", original_shift
                )
                pipe.scheduler = new_scheduler
            elif scheduler == "Karras Euler":
                new_scheduler = get_scheduler_instance(
                    "FlashFlowMatchEulerDiscreteScheduler", original_shift
                )
                if hasattr(new_scheduler, "use_karras_sigmas"):
                    new_scheduler.use_karras_sigmas = True
                pipe.scheduler = new_scheduler
            elif scheduler == "Karras Exponential":
                new_scheduler = get_scheduler_instance(
                    "FlashFlowMatchEulerDiscreteScheduler", original_shift
                )
                if hasattr(new_scheduler, "use_exponential_sigmas"):
                    new_scheduler.use_exponential_sigmas = True
                pipe.scheduler = new_scheduler
        else:
            # Ensure we're using the original scheduler as specified in the model config
            print(f"Using model's default scheduler: {original_scheduler_class}")
            pipe.scheduler = get_scheduler_instance(
                original_scheduler_class, original_shift
            )

        # Setup generation parameters
        is_nf4_current = config.get("is_nf4", False)
        num_inference_steps = (
            override_steps if override_steps >= 0 else config["num_inference_steps"]
        )
        guidance_scale = (
            override_cfg if override_cfg >= 0.0 else config["guidance_scale"]
        )

        # Create progress bar
        pbar = comfy.utils.ProgressBar(num_inference_steps)

        # Define progress callback
        def progress_callback(pipe, i, t, callback_kwargs):
            # Update ComfyUI progress bar
            pbar.update_absolute(i + 1)
            return callback_kwargs

        try:
            inference_device = comfy.model_management.get_torch_device()
        except Exception:
            inference_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        print(f"Creating Generator on: {inference_device}")
        generator = torch.Generator(device=inference_device).manual_seed(seed)
        print("\n--- Starting Img2Img Generation ---")
        _, h, w, _ = image.shape
        print(
            f"Model: {model_type}{' (uncensored)' if use_uncensored_llm else ''}, Input Size: {h}x{w}"
        )
        print(
            f"Denoising: {denoising_strength}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}"
        )

        # --- Run Inference ---
        output_images = None
        try:
            if not is_nf4_current:
                print(f"Ensuring pipe on: {inference_device} (Offload NOT enabled)")
                pipe.to(inference_device)
            else:
                print(f"Skipping pipe.to({inference_device}) (CPU offload enabled).")

            print("Executing pipeline inference...")

            with torch.inference_mode():
                output_images = pipe(
                    prompt=prompt,
                    prompt_2=prompt,  # Same prompt for all encoders
                    prompt_3=prompt,
                    prompt_4=prompt,
                    negative_prompt=negative_prompt.strip()
                    if negative_prompt
                    else None,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    init_image=processed_image,
                    denoising_strength=denoising_strength,
                    llm_system_prompt=llm_system_prompt,
                    clip_l_scale=clip_l_weight,
                    openclip_scale=openclip_weight,
                    t5_scale=t5_weight,
                    llama_scale=llama_weight,
                    callback_on_step_end=progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images

            print("Pipeline inference finished.")

        except Exception as e:
            print(f"!!! ERROR during execution: {e}")
            import traceback

            traceback.print_exc()
            return (torch.zeros((1, h, w, 3)),)
        finally:
            pbar.update_absolute(num_inference_steps)  # Update pbar regardless
        print("--- Generation Complete ---")

        # Robust output handling
        if output_images is None or len(output_images) == 0:
            print("ERROR: No images returned. Creating blank image.")
            return (torch.zeros((1, h, w, 3)),)

        try:
            print(f"Processing output image. Type: {type(output_images[0])}")
            output_tensor = pil2tensor(output_images[0])

            if output_tensor is None:
                print("ERROR: pil2tensor returned None. Creating blank image.")
                return (torch.zeros((1, h, w, 3)),)

            # Fix for bfloat16 tensor issue
            if output_tensor.dtype == torch.bfloat16:
                print("Converting bfloat16 tensor to float32 for ComfyUI compatibility")
                output_tensor = output_tensor.to(torch.float32)

            # Verify tensor shape is valid
            if (
                len(output_tensor.shape) != 4
                or output_tensor.shape[0] != 1
                or output_tensor.shape[3] != 3
            ):
                print(
                    f"ERROR: Invalid tensor shape {output_tensor.shape}. Creating blank image."
                )
                return (torch.zeros((1, h, w, 3)),)

            print(f"Output tensor shape: {output_tensor.shape}")

            # After generating the image, try to clean up any temporary memory
            try:
                import comfy.model_management as model_management

                print("HiDream: Requesting ComfyUI memory cleanup...")
                model_management.soft_empty_cache()
            except Exception as e:
                print(f"HiDream: ComfyUI cleanup failed: {e}")

            # Log final memory usage
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated() / 1024**2
                print(
                    f"HiDream: Final VRAM usage: {final_mem:.2f} MB (Change: {final_mem - initial_mem:.2f} MB)"
                )

            return (output_tensor,)

        except Exception as e:
            print(f"Error processing output image: {e}")
            import traceback

            traceback.print_exc()
            return (torch.zeros((1, h, w, 3)),)
