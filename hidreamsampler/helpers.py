from .config import (
    MODEL_CONFIGS,
    available_schedulers,
    hidream_classes_loaded,
    bnb_available,
    optimum_available,
    gptqmodel_available,
    bnb_llm_config,
    bnb_transformer_4bit_config,
    model_dtype,
    HiDreamImageTransformer2DModel,
    HiDreamImagePipeline,
    HiDreamImageToImagePipeline,
    FlowUniPCMultistepScheduler,
    FlashFlowMatchEulerDiscreteScheduler,
    DEBUG_CACHE,
    UNCENSORED_NF4_LLAMA_MODEL_NAME,
    NF4_LLAMA_MODEL_NAME,
    UNCENSORED_LLAMA_MODEL_NAME,
    ORIGINAL_LLAMA_MODEL_NAME,
)
import torch
import numpy as np
from PIL import Image
import gc
import comfy.utils


# Use a more aggressive global cleanup
def global_cleanup():
    """Global cleanup function for use with multiple HiDream nodes"""
    print("HiDream: Performing global cleanup...")

    # Clear any pending operations
    torch.cuda.synchronize()

    # Get current memory stats
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  Memory before cleanup: {before_mem:.2f} MB")

    # Perform HiDreamSampler cleanup
    from .nodes import HiDreamSampler

    HiDreamSampler.cleanup_models()

    # Additional cleanup
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  Memory after cleanup: {after_mem:.2f} MB")

    return True


# --- Helper: Get Scheduler Instance ---
def get_scheduler_instance(scheduler_name, shift_value):
    if not available_schedulers:
        raise RuntimeError("No schedulers available...")
    scheduler_class = available_schedulers.get(scheduler_name)
    if scheduler_class is None:
        raise ValueError(f"Scheduler class '{scheduler_name}' not found...")
    return scheduler_class(
        num_train_timesteps=1000, shift=shift_value, use_dynamic_shifting=False
    )


# --- Loading Function (Handles NF4 and default BNB) ---
def load_models(model_type, use_uncensored_llm=False):
    if not hidream_classes_loaded:
        raise ImportError("Cannot load models: HiDream classes failed to import.")
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown or incompatible model_type: {model_type}")
    config = MODEL_CONFIGS[model_type]
    model_path = config["path"]
    is_nf4 = config.get("is_nf4", False)
    scheduler_name = config["scheduler_class"]
    shift = config["shift"]
    requires_bnb = config.get("requires_bnb", False)
    requires_gptq_deps = config.get("requires_gptq_deps", False)
    if requires_bnb and not bnb_available:
        raise ImportError(f"Model '{model_type}' requires BitsAndBytes...")
    if requires_gptq_deps and (not optimum_available or not gptqmodel_available):
        raise ImportError(f"Model '{model_type}' requires Optimum & AutoGPTQ...")
    print(f"--- Loading Model Type: {model_type} ---")
    print(f"Model Path: {model_path}")
    print(
        f"NF4: {is_nf4}, Requires BNB: {requires_bnb}, Requires GPTQ deps: {requires_gptq_deps}"
    )
    print(f"Using Uncensored LLM: {use_uncensored_llm}")
    start_mem = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    )
    print(f"(Start VRAM: {start_mem:.2f} MB)")

    # Create a standardized cache key used by all nodes
    from .nodes import HiDreamSampler

    cache_key = f"{model_type}_{'uncensored' if use_uncensored_llm else 'standard'}"

    # Check cache with debug info
    if DEBUG_CACHE:
        print(f"Cache check for key: {cache_key}")
        print(f"Cache contains: {list(HiDreamSampler._model_cache.keys())}")

    if cache_key in HiDreamSampler._model_cache:
        pipe, stored_config = HiDreamSampler._model_cache[cache_key]
        if (
            pipe is not None
            and hasattr(pipe, "transformer")
            and pipe.transformer is not None
        ):
            print(f"Using cached model for {cache_key}")
            return pipe, MODEL_CONFIGS[model_type]  # Always return original config dict
        else:
            print(f"Cache entry invalid for {cache_key}, reloading")
            # Remove from cache to avoid reusing
            HiDreamSampler._model_cache.pop(cache_key, None)

    # --- 1. Load LLM (Conditional) ---
    text_encoder_load_kwargs = {"low_cpu_mem_usage": True, "torch_dtype": model_dtype}

    if is_nf4:
        # Choose uncensored model if requested, but keep loading process identical
        if use_uncensored_llm:
            llama_model_name = UNCENSORED_NF4_LLAMA_MODEL_NAME
            print(f"\n[1a] Preparing Uncensored LLM (GPTQ): {llama_model_name}")
        else:
            llama_model_name = NF4_LLAMA_MODEL_NAME
            print(f"\n[1a] Preparing LLM (GPTQ): {llama_model_name}")

        # Rest of the NF4 loading process stays exactly the same
        from .config import accelerate_available

        if accelerate_available:
            # Fix for device format - use integer instead of cuda:0
            if (
                hasattr(torch.cuda, "get_device_properties")
                and torch.cuda.is_available()
            ):
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Use 40% for model, leaving room for the transformer
                max_mem = int(total_mem * 0.4)
                text_encoder_load_kwargs["max_memory"] = {0: f"{max_mem}GiB"}
                print(
                    f"     Setting max memory limit: {max_mem}GiB of {total_mem:.1f}GiB"
                )
            text_encoder_load_kwargs["device_map"] = "auto"
            print("     Using device_map='auto'.")
        else:
            print("     accelerate not found, attempting manual placement.")
    else:
        # For non-NF4 models, choose uncensored if requested
        if use_uncensored_llm:
            llama_model_name = UNCENSORED_LLAMA_MODEL_NAME
            print(f"\n[1a] Preparing Uncensored LLM (4-bit BNB): {llama_model_name}")
        else:
            llama_model_name = ORIGINAL_LLAMA_MODEL_NAME
            print(f"\n[1a] Preparing LLM (4-bit BNB): {llama_model_name}")

        # Rest of standard model loading stays exactly the same
        if bnb_llm_config:
            text_encoder_load_kwargs["quantization_config"] = bnb_llm_config
            print("     Using 4-bit BNB.")
        else:
            raise ImportError("BNB config required for standard LLM.")

    from transformers import LlamaForCausalLM, AutoTokenizer

    print(f"[1b] Loading Tokenizer: {llama_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False)
    print("     Tokenizer loaded.")

    if is_nf4:
        # More aggressive RoPE scaling fix
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(llama_model_name)
            # config.rope_scaling = {"type": "linear", "factor": 1.0}
            # print(f"     ✅ Fixed rope_scaling to: {config.rope_scaling}")
            text_encoder_load_kwargs["config"] = config
            text_encoder_load_kwargs["low_cpu_mem_usage"] = True
        except Exception as e:
            print(f"     ⚠️ Failed to patch config: {e}")

    print(f"[1c] Loading Text Encoder: {llama_model_name}... (May download files)")
    text_encoder = LlamaForCausalLM.from_pretrained(
        llama_model_name, **text_encoder_load_kwargs
    )
    if "device_map" not in text_encoder_load_kwargs:
        print("     Moving text encoder to CUDA...")
        text_encoder.to("cuda")
    step1_mem = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    )
    print(f"✅ Text encoder loaded! (VRAM: {step1_mem:.2f} MB)")

    # --- 2. Load Transformer (Conditional) ---
    print(f"\n[2] Preparing Transformer from: {model_path}")
    transformer_load_kwargs = {
        "subfolder": "transformer",
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    if is_nf4:
        print("     Type: NF4")
    else:  # Default BNB case
        print("     Type: Standard (Applying 4-bit BNB quantization)")
        if bnb_transformer_4bit_config:
            transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config
        else:
            raise ImportError("BNB config required for transformer but unavailable.")
    print("     Loading Transformer... (May download files)")
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_path, **transformer_load_kwargs
    )
    print("     Moving Transformer to CUDA...")
    transformer.to("cuda")
    step2_mem = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    )
    print(f"✅ Transformer loaded! (VRAM: {step2_mem:.2f} MB)")

    # --- 3. Load Scheduler ---
    print(f"\n[3] Preparing Scheduler: {scheduler_name}")
    scheduler = get_scheduler_instance(scheduler_name, shift)
    print(f"     Using Scheduler: {scheduler_name}")

    # --- 4. Load Pipeline ---
    print(f"\n[4] Loading Pipeline from: {model_path}")
    print("     Passing pre-loaded components...")
    pipe = HiDreamImagePipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        transformer=None,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    print("     Pipeline structure loaded.")

    # --- 5. Final Setup ---
    print("\n[5] Finalizing Pipeline...")
    print("     Assigning transformer...")
    pipe.transformer = transformer
    print("     Moving pipeline object to CUDA (final check)...")
    try:
        pipe.to("cuda")
    except Exception as e:
        print(f"     Warning: Could not move pipeline object to CUDA: {e}.")
    if is_nf4:
        print("     Attempting CPU offload for NF4...")
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                pipe.enable_sequential_cpu_offload()
                print("     ✅ CPU offload enabled.")
            except Exception as e:
                print(f"     ⚠️ Failed CPU offload: {e}")
        else:
            print("     ⚠️ enable_sequential_cpu_offload() not found.")
    final_mem = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    )
    print(f"✅ Pipeline ready! (VRAM: {final_mem:.2f} MB)")
    return pipe, MODEL_CONFIGS[model_type]


# --- Resolution Parsing & Tensor Conversion ---
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)",
]


def parse_resolution(resolution_str):
    """Parse resolution string into height and width dimensions."""
    try:
        # Extract the resolution part before the parenthesis
        res_part = resolution_str.split(" (")[0].strip()
        # Replace 'x' with '×' for consistency if needed
        parts = res_part.replace("x", "×").split("×")

        if len(parts) != 2:
            raise ValueError(f"Expected format 'width × height', got '{res_part}'")

        width_str = parts[0].strip()
        height_str = parts[1].strip()

        width = int(width_str)
        height = int(height_str)
        print(f"Successfully parsed resolution: {width}x{height}")
        return height, width
    except Exception as e:
        print(
            f"Error parsing resolution '{resolution_str}': {e}. Falling back to 1024x1024."
        )
        return 1024, 1024


def pil2tensor(image: Image.Image):
    """Convert PIL image to tensor with better error handling"""
    if image is None:
        print("pil2tensor: Image is None")
        return None

    try:
        # Debug image properties
        print(f"pil2tensor: Image mode={image.mode}, size={image.size}")

        # Ensure image is in RGB mode
        if image.mode != "RGB":
            print(f"Converting image from {image.mode} to RGB")
            image = image.convert("RGB")

        # Convert to numpy array with explicit steps
        np_array = np.array(image)
        print(f"Numpy array shape={np_array.shape}, dtype={np_array.dtype}")

        # Convert to float32 and normalize
        np_array = np_array.astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(np_array)
        tensor = tensor.unsqueeze(0)
        print(f"Final tensor shape={tensor.shape}")

        return tensor
    except Exception as e:
        print(f"Error in pil2tensor: {e}")
        import traceback

        traceback.print_exc()

        # Try ComfyUI's own conversion if ours fails
        try:
            print("Trying ComfyUI's own conversion...")
            tensor = comfy.utils.pil2tensor(image)
            print(f"ComfyUI conversion successful: {tensor.shape}")
            return tensor
        except Exception as e2:
            print(f"ComfyUI conversion also failed: {e2}")
            return None
