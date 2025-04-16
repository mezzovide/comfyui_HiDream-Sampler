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

from .hidream_base import HiDreamBase


class HiDreamSamplerAdvanced(HiDreamBase):
    _model_cache = HiDreamBase._model_cache
    cleanup_models = HiDreamBase.cleanup_models

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
                "primary_prompt": ("STRING", {"multiline": True, "default": "..."}),
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
                "use_uncensored_llm": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "clip_l_prompt": ("STRING", {"multiline": True, "default": ""}),
                "openclip_prompt": ("STRING", {"multiline": True, "default": ""}),
                "t5_prompt": ("STRING", {"multiline": True, "default": ""}),
                "llama_prompt": ("STRING", {"multiline": True, "default": ""}),
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
                "max_length_clip_l": ("INT", {"default": 77, "min": 64, "max": 218}),
                "max_length_openclip": ("INT", {"default": 77, "min": 64, "max": 218}),
                "max_length_t5": ("INT", {"default": 128, "min": 64, "max": 512}),
                "max_length_llama": ("INT", {"default": 128, "min": 64, "max": 2048}),
            },
        }

    def generate(
        self,
        model_type,
        primary_prompt,
        negative_prompt,
        resolution,
        num_images,
        seed,
        scheduler,
        override_steps,
        override_cfg,
        use_uncensored_llm=False,
        clip_l_prompt="",
        openclip_prompt="",
        t5_prompt="",
        llama_prompt="",
        llm_system_prompt="You are a creative AI assistant...",
        override_width=0,
        override_height=0,
        max_length_clip_l=77,
        max_length_openclip=77,
        max_length_t5=128,
        max_length_llama=128,
        clip_l_weight=1.0,
        openclip_weight=1.0,
        t5_weight=1.0,
        llama_weight=1.0,
        **kwargs,
    ):
        # Determine resolution
        if override_width > 0 and override_height > 0:
            height, width = override_height, override_width
            print(f"Using override resolution: {width}x{height}")
        else:
            height, width = parse_resolution(resolution)
            print(f"Using fixed resolution: {width}x{height} ({resolution})")

        # Monitor initial memory usage
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1024**2
            print(f"HiDream: Initial VRAM usage: {initial_mem:.2f} MB")

        if not MODEL_CONFIGS or model_type == "error":
            print("HiDream Error: No models loaded.")
            return (torch.zeros((1, 512, 512, 3)),)

        pipe = None
        config = None
        cache_key = f"{model_type}_{'uncensored' if use_uncensored_llm else 'standard'}"

        # --- Model Loading / Caching ---
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
                print(f"Clearing ALL cache before loading {model_type}...")
                keys_to_del = list(self._model_cache.keys())
                for key in keys_to_del:
                    print(f"  Removing '{key}'...")
                    try:
                        pipe_to_del, _ = self._model_cache.pop(key)
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

            print(
                f"Loading model for {model_type}{' (uncensored)' if use_uncensored_llm else ''}..."
            )

            try:
                pipe, config = load_models(model_type, use_uncensored_llm)
                self._model_cache[cache_key] = (pipe, config)
                print(
                    f"Model {model_type}{' (uncensored)' if use_uncensored_llm else ''} loaded & cached!"
                )
            except Exception as e:
                print(f"!!! ERROR loading {model_type}: {e}")
                import traceback

                traceback.print_exc()
                return (torch.zeros((1, 512, 512, 3)),)

        if pipe is None or config is None:
            print("CRITICAL ERROR: Load failed.")
            return (torch.zeros((1, 512, 512, 3)),)

        # --- Update scheduler if requested ---
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

        # --- Generation Setup ---
        is_nf4_current = config.get("is_nf4", False)
        num_inference_steps = (
            override_steps if override_steps >= 0 else config["num_inference_steps"]
        )
        guidance_scale = (
            override_cfg if override_cfg >= 0.0 else config["guidance_scale"]
        )

        # Create the progress bar
        pbar = comfy.utils.ProgressBar(num_inference_steps)

        try:
            inference_device = comfy.model_management.get_torch_device()
        except Exception:
            inference_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        print(f"Creating Generator on: {inference_device}")
        generator = torch.Generator(device=inference_device).manual_seed(seed)
        print("\n--- Starting Generation ---")
        print(
            f"Model: {model_type}{' (uncensored)' if use_uncensored_llm else ''}, Res: {height}x{width}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}"
        )
        print(
            f"Sequence lengths - CLIP-L: {max_length_clip_l}, OpenCLIP: {max_length_openclip}, T5: {max_length_t5}, Llama: {max_length_llama}"
        )

        # --- Run Inference ---
        pipeline_output = None
        try:
            if not is_nf4_current:
                print(f"Ensuring pipe on: {inference_device} (Offload NOT enabled)")
                pipe.to(inference_device)
            else:
                print(f"Skipping pipe.to({inference_device}) (CPU offload enabled).")

            print("Executing pipeline inference...")

            # Use specific prompts for each encoder, falling back to primary prompt if empty
            prompt_clip_l = (
                clip_l_prompt.strip() if clip_l_prompt.strip() else primary_prompt
            )
            prompt_openclip = (
                openclip_prompt.strip() if openclip_prompt.strip() else primary_prompt
            )
            prompt_t5 = t5_prompt.strip() if t5_prompt.strip() else primary_prompt
            prompt_llama = (
                llama_prompt.strip() if llama_prompt.strip() else primary_prompt
            )

            print("Using per-encoder prompts:")
            print(
                f"  CLIP-L ({max_length_clip_l} tokens): {prompt_clip_l[:50]}{'...' if len(prompt_clip_l) > 50 else ''}"
            )
            print(
                f"  OpenCLIP ({max_length_openclip} tokens): {prompt_openclip[:50]}{'...' if len(prompt_openclip) > 50 else ''}"
            )
            print(
                f"  T5 ({max_length_t5} tokens): {prompt_t5[:50]}{'...' if len(prompt_t5) > 50 else ''}"
            )
            print(
                f"  Llama ({max_length_llama} tokens): {prompt_llama[:50]}{'...' if len(prompt_llama) > 50 else ''}"
            )

            # Replace truly blank inputs with minimal period
            if not prompt_clip_l.strip():
                prompt_clip_l = "."

            if not prompt_openclip.strip():
                prompt_openclip = "."

            if not prompt_t5.strip():
                prompt_t5 = "."

            # Custom system prompt for blank LLM prompts to try to prevent LLM output noise
            custom_system_prompt = llm_system_prompt
            if not prompt_llama.strip():
                prompt_llama = "."
                custom_system_prompt = "You will only output a single period as your output '.'\nDo not add any other acknowledgement or extra text or data."

            # Ensure batch size consistency for multiple images
            if num_images > 1:
                print(f"Preparing for batch generation with {num_images} images...")
                # Create a list to store outputs
                output_images_list = []
                for i in range(num_images):
                    print(f"Generating image {i + 1}/{num_images}...")
                    # Generate one image at a time to avoid batch size issues
                    with torch.inference_mode():
                        single_output = pipe(
                            prompt=prompt_clip_l,
                            prompt_2=prompt_openclip,
                            prompt_3=prompt_t5,
                            prompt_4=prompt_llama,
                            negative_prompt=negative_prompt.strip()
                            if negative_prompt
                            else None,
                            height=height,
                            width=width,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            num_images_per_prompt=1,
                            generator=generator,
                            max_sequence_length_clip_l=max_length_clip_l,
                            max_sequence_length_openclip=max_length_openclip,
                            max_sequence_length_t5=max_length_t5,
                            max_sequence_length_llama=max_length_llama,
                            llm_system_prompt=custom_system_prompt,
                            clip_l_scale=clip_l_weight,
                            openclip_scale=openclip_weight,
                            t5_scale=t5_weight,
                            llama_scale=llama_weight,
                            callback_on_step_end_tensor_inputs=["latents"],
                        )
                    output_images_list.extend(single_output.images)
                    pbar.update_absolute((i + 1) * num_inference_steps // num_images)
            else:
                with torch.inference_mode():
                    pipeline_output = pipe(
                        prompt=prompt_clip_l,
                        prompt_2=prompt_openclip,
                        prompt_3=prompt_t5,
                        prompt_4=prompt_llama,
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
                        llm_system_prompt=custom_system_prompt,
                        clip_l_scale=clip_l_weight,
                        openclip_scale=openclip_weight,
                        t5_scale=t5_weight,
                        llama_scale=llama_weight,
                        callback_on_step_end_tensor_inputs=["latents"],
                    )
                    output_images_list = pipeline_output.images

            print("Pipeline inference finished.")

        except Exception as e:
            print(f"!!! ERROR during execution: {e}")
            import traceback

            traceback.print_exc()
            return (torch.zeros((1, height, width, 3)),)
        finally:
            pbar.update_absolute(num_inference_steps)  # Update pbar regardless
        print("--- Generation Complete ---")

        # Robust output handling
        if (
            output_images_list is None
            or not isinstance(output_images_list, list)
            or len(output_images_list) == 0
        ):
            print(
                f"ERROR: No images returned or invalid format (Type: {type(output_images_list)}). Creating blank image."
            )
            return (torch.zeros((1, height, width, 3)),)

        try:
            print(f"Processing {len(output_images_list)} output image(s).")
            tensor_list = []

            for i, img in enumerate(output_images_list):
                if not isinstance(img, Image.Image):
                    print(
                        f"WARNING: Item {i} in output list is not a PIL Image (Type: {type(img)}). Skipping."
                    )
                    continue

                print(f"Converting image {i + 1}/{len(output_images_list)}...")
                single_tensor = pil2tensor(img)  # This returns shape [1, H, W, C]

                if single_tensor is not None:
                    if len(single_tensor.shape) == 4 and single_tensor.shape[0] == 1:
                        tensor_list.append(single_tensor)
                    else:
                        print(
                            f"WARNING: pil2tensor returned unexpected shape {single_tensor.shape} for image {i}. Skipping."
                        )
                else:
                    print(f"WARNING: pil2tensor failed for image {i}. Skipping.")

            if not tensor_list:
                print("ERROR: All image conversions failed. Creating blank image.")
                return (torch.zeros((1, height, width, 3)),)

            output_tensor = torch.cat(tensor_list, dim=0)
            print(
                f"Successfully converted {output_tensor.shape[0]} images into batch tensor."
            )

            if output_tensor.dtype != torch.float32:
                print(
                    f"Converting batched {output_tensor.dtype} tensor to float32 for ComfyUI compatibility"
                )
                output_tensor = output_tensor.to(torch.float32)

            if (
                len(output_tensor.shape) != 4
                or output_tensor.shape[0] == 0
                or output_tensor.shape[3] != 3
            ):
                print(
                    f"ERROR: Invalid final batch tensor shape {output_tensor.shape}. Creating blank image."
                )
                return (torch.zeros((1, height, width, 3)),)

            print(f"Output tensor shape: {output_tensor.shape}")

            try:
                import comfy.model_management as model_management

                print("HiDream: Requesting ComfyUI memory cleanup...")
                model_management.soft_empty_cache()
            except Exception as e:
                print(f"HiDream: ComfyUI cleanup failed: {e}")

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
            return (torch.zeros((1, height, width, 3)),)
