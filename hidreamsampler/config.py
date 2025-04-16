# -*- coding: utf-8 -*-
# HiDream Sampler Node for ComfyUI
# Version: 2024-07-29d (Removed auto-gptq dependency check)
#
# Required Dependencies:
# - transformers, diffusers, torch, numpy, Pillow
# - For NF4 (GPTQ) models: optimum, accelerate (`pip install optimum accelerate`)
#   * Note: Optimum might require additional backends like exllama (`pip install optimum[exllama]`) depending on your setup.
# - For non-NF4/FP8 models (4-bit): bitsandbytes (`pip install bitsandbytes`)
# - Ensure hi_diffusers library is locally available or hdi1 package is installed.

import torch
import importlib.util
import logging

logger = logging.getLogger(__name__)

accelerate_spec = importlib.util.find_spec("accelerate")
accelerate_available = accelerate_spec is not None
if not accelerate_available:
    logger.warning(
        "Warning: accelerate not installed. device_map='auto' for GPTQ models may not work optimally."
    )

gptqmodel_spec = importlib.util.find_spec("gptqmodel")
gptqmodel_available = gptqmodel_spec is not None
if not gptqmodel_available:
    logger.warning("Warning: GPTQModel not installed.")
    # Note: Optimum might still load GPTQ without GPTQModel if using ExLlama kernels,
    # but it's often required. Add a warning if NF4 models are selected later.^
optimum_spec = importlib.util.find_spec("optimum")
optimum_available = optimum_spec is not None
if optimum_available:
    logger.info(
        "Optimum library found. GPTQ model loading enabled (requires suitable backend)."
    )
else:
    logger.warning(
        "Warning: optimum not installed. GPTQ models (NF4 variants) will be disabled."
    )

try:
    # Import specific classes to avoid potential namespace conflicts later
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

    bnb_available = True
except ImportError:
    bnb_available = False
    # Keep placeholders None to avoid errors later if bnb not available
    TransformersBitsAndBytesConfig = None
    DiffusersBitsAndBytesConfig = None
    logger.warning(
        "Warning: bitsandbytes not installed. 4-bit BNB quantization will not be available."
    )

# --- HiDream Specific Imports ---
# Attempt local import first, then fallback (which might fail)
try:
    # Assuming hi_diffusers is cloned into this custom_node's directory
    from ..hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from ..hi_diffusers.schedulers.flash_flow_match import (
        FlashFlowMatchEulerDiscreteScheduler,
    )
    from ..hi_diffusers.models.transformers.transformer_hidream_image import (
        HiDreamImageTransformer2DModel,
    )
    from ..hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import (
        HiDreamImagePipeline,
    )
    from ..hi_diffusers.pipelines.hidream_image.pipeline_hidream_image_to_image import (
        HiDreamImageToImagePipeline,
    )

    hidream_classes_loaded = True
except ImportError as e:
    logger.error("--------------------------------------------------------------------")
    logger.error(f"ComfyUI-HiDream-Sampler: Could not import local hi_diffusers ({e}).")
    logger.error(
        "Please ensure hi_diffusers library is inside ComfyUI-HiDream-Sampler,"
    )
    logger.error("or hdi1 package is installed in the ComfyUI environment.")
    logger.error("Node may fail to load models.")
    logger.error("--------------------------------------------------------------------")
    # Define placeholders so the script doesn't crash immediately
    HiDreamImageTransformer2DModel = None
    HiDreamImagePipeline = None
    FlowUniPCMultistepScheduler = None
    FlashFlowMatchEulerDiscreteScheduler = None
    hidream_classes_loaded = False

# --- Model Paths ---
ORIGINAL_MODEL_PREFIX = "HiDream-ai"
NF4_MODEL_PREFIX = "azaneko"

ORIGINAL_LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
NF4_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
UNCENSORED_LLAMA_MODEL_NAME = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
UNCENSORED_NF4_LLAMA_MODEL_NAME = "shuttercat/DarkIdol-Llama3.1-NF4-GPTQ"

# --- Model Configurations ---
MODEL_CONFIGS = {
    # --- NF4 Models (Require Optimum) ---
    "full-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": True,
        "is_fp8": False,
        "requires_bnb": False,
        "requires_gptq_deps": True,  # Requires Optimum
    },
    "dev-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True,
        "is_fp8": False,
        "requires_bnb": False,
        "requires_gptq_deps": True,  # Requires Optimum
    },
    "fast-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True,
        "is_fp8": False,
        "requires_bnb": False,
        "requires_gptq_deps": True,  # Requires Optimum
    },
    # --- Original/BNB Models (Require BitsAndBytes) ---
    "full": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False,
        "is_fp8": False,
        "requires_bnb": True,
        "requires_gptq_deps": False,
    },
    "dev": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False,
        "is_fp8": False,
        "requires_bnb": True,
        "requires_gptq_deps": False,
    },
    "fast": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False,
        "is_fp8": False,
        "requires_bnb": True,
        "requires_gptq_deps": False,
    },
}

RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)",
]

# --- Filter models based on available dependencies ---
original_model_count = len(MODEL_CONFIGS)
if not bnb_available:
    MODEL_CONFIGS = {
        k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_bnb", False)
    }
if not optimum_available or not gptqmodel_available:
    MODEL_CONFIGS = {
        k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_gptq_deps", False)
    }
if not hidream_classes_loaded:
    MODEL_CONFIGS = {}
filtered_model_count = len(MODEL_CONFIGS)
if filtered_model_count == 0:
    logger.error(
        "*" * 70 + "\nCRITICAL ERROR: No HiDream models available...\n" + "*" * 70
    )
elif filtered_model_count < original_model_count:
    logger.warning("*" * 70 + "\nWarning: Some HiDream models disabled...\n" + "*" * 70)

# Define BitsAndBytes configs (if available)
bnb_llm_config = None
bnb_transformer_4bit_config = None
if bnb_available:
    bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
    bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)

model_dtype = torch.bfloat16

# Get available scheduler classes
available_schedulers = {}
if hidream_classes_loaded:
    available_schedulers = {
        "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler,
        "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler,
    }

DEBUG_CACHE = True  # Set to False in production
