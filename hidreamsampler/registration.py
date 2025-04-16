from .nodes.hidream_sampler import HiDreamSampler
from .nodes.hidream_img2img import HiDreamImg2Img
from .nodes.hidream_sampler_advanced import HiDreamSamplerAdvanced
from .config import MODEL_CONFIGS
import logging

logger = logging.getLogger("hidreamsampler")

NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler,
    "HiDreamSamplerAdvanced": HiDreamSamplerAdvanced,
    "HiDreamImg2Img": HiDreamImg2Img,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler",
    "HiDreamSamplerAdvanced": "HiDream Sampler (Advanced)",
    "HiDreamImg2Img": "HiDream Image to Image",
}

# --- Register with ComfyUI's Memory Management ---
try:
    import comfy.model_management as model_management

    # Check if we can register a cleanup callback
    if hasattr(model_management, "unload_all_models"):
        original_unload = model_management.unload_all_models

        # Wrap the original function to include our cleanup
        def wrapped_unload():
            logger.info(
                "HiDream: ComfyUI is unloading all models, cleaning HiDream cache..."
            )
            HiDreamSampler.cleanup_models()
            return original_unload()

        # Replace the original function with our wrapped version
        model_management.unload_all_models = wrapped_unload
        logger.info("HiDream: Successfully registered with ComfyUI memory management")
except Exception as e:
    logger.warning(f"HiDream: Could not register cleanup with model_management: {e}")

logger.info(
    "-" * 50
    + "\nHiDream Sampler Node Initialized\nAvailable Models: "
    + str(list(MODEL_CONFIGS.keys()))
    + "\n"
    + "-" * 50
)
