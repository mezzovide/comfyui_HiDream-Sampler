import logging

# Set up logging for this module
logger = logging.getLogger(__name__)

import torch
import gc
from ..helpers import (
    get_scheduler_instance,
    pil2tensor,
)


class HiDreamBase:
    _model_cache = {}

    @classmethod
    def cleanup_models(cls):
        logger.info("HiDream: Cleaning up all cached models...")
        keys_to_del = list(cls._model_cache.keys())
        for key in keys_to_del:
            logger.info("Removing '{}'...".format(key))
            try:
                pipe_to_del, _ = cls._model_cache.pop(key)
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
                logger.error("Error cleaning up {}: {}".format(key, e))
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("HiDream: Cache cleared")
        return True

    @staticmethod
    def get_model_from_cache(cache_key, model_cache):
        if cache_key in model_cache:
            logger.info("Checking cache for {}...".format(cache_key))
            pipe, config = model_cache[cache_key]
            valid_cache = True
            if (
                pipe is None
                or config is None
                or not hasattr(pipe, "transformer")
                or pipe.transformer is None
            ):
                valid_cache = False
                logger.warning("Invalid cache, reloading...")
                del model_cache[cache_key]
                pipe, config = None, None
            if valid_cache:
                logger.info("Using cached model.")
            return pipe, config
        return None, None

    @staticmethod
    def clear_model_cache(model_cache):
        if model_cache:
            logger.info("Clearing ALL cache before loading new model...")
            keys_to_del = list(model_cache.keys())
            for key in keys_to_del:
                logger.info("Removing '{}'...".format(key))
                try:
                    pipe_to_del, _ = model_cache.pop(key)
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
                    logger.error("Error removing {}: {}".format(key, e))
            for _ in range(3):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("Cache cleared.")

    @staticmethod
    def update_scheduler(
        pipe, config, scheduler, original_scheduler_class, original_shift
    ):
        if scheduler != "Default for model":
            logger.info(
                "Replacing default scheduler ({}) with: {}".format(
                    original_scheduler_class, scheduler
                )
            )
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
            logger.info(
                "Using model's default scheduler: {}".format(original_scheduler_class)
            )
            pipe.scheduler = get_scheduler_instance(
                original_scheduler_class, original_shift
            )

    @staticmethod
    def process_output_images(output_images, h, w):
        if output_images is None or len(output_images) == 0:
            logger.error("No images returned. Creating blank image.")
            return (torch.zeros((1, h, w, 3)),)
        try:
            logger.info(
                "Processing output image. Type: {}".format(type(output_images[0]))
            )
            output_tensor = pil2tensor(output_images[0])
            if output_tensor is None:
                logger.error("pil2tensor returned None. Creating blank image.")
                return (torch.zeros((1, h, w, 3)),)
            if output_tensor.dtype == torch.bfloat16:
                logger.info(
                    "Converting bfloat16 tensor to float32 for ComfyUI compatibility"
                )
                output_tensor = output_tensor.to(torch.float32)
            if (
                len(output_tensor.shape) != 4
                or output_tensor.shape[0] != 1
                or output_tensor.shape[3] != 3
            ):
                logger.error(
                    "Invalid tensor shape {}. Creating blank image.".format(
                        output_tensor.shape
                    )
                )
                return (torch.zeros((1, h, w, 3)),)
            logger.info("Output tensor shape: {}".format(output_tensor.shape))
            try:
                import comfy.model_management as model_management

                logger.info("HiDream: Requesting ComfyUI memory cleanup...")
                model_management.soft_empty_cache()
            except Exception as e:
                logger.warning("HiDream: ComfyUI cleanup failed: {}".format(e))
            return (output_tensor,)
        except Exception as e:
            logger.error("Error processing output image: {}".format(e))
            import traceback

            traceback.print_exc()
            return (torch.zeros((1, h, w, 3)),)
