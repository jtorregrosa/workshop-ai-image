from diffusers import FluxPipeline, StableDiffusionPipeline, FluxImg2ImgPipeline, FluxControlPipeline
import torch

def load_flux_model():
    try:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load Flux.1 model: {e}")

def load_flux_model_lora_txt2img():
    try:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        # You should use "ghibli" to trigger the image generation.
        pipe.load_lora_weights('openfree/flux-chatgpt-ghibli-lora', weight_name='flux-chatgpt-ghibli-lora.safetensors')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load Flux.1 model: {e}")
    
def load_sd35_model():
    try:
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load Stable Diffusion 3.5 model: {e}")

def load_flux_model_img2img():
    try:
        pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load Flux.1 model: {e}")
