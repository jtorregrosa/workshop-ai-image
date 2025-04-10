from PIL import Image
from controlnet_aux import CannyDetector
from model import load_flux_model, load_sd35_model, load_flux_model_lora_txt2img, load_flux_model_img2img
from model_manager import ModelManager

manager = ModelManager({
    'flux_txt2img': load_flux_model, 
    'flux_img2img': load_flux_model_img2img,
    'flux_lora_txt2img': load_flux_model_lora_txt2img,
    'sd35_txt2img': load_sd35_model,
})

def generate_output_flux_txt2img(prompt: str, width: int, height: int, steps: int):
    try:
        result = manager.get_model('flux_txt2img')(
            prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            output_type="pil",
            num_inference_steps=steps,
            max_sequence_length=512,
        )
        return result.images[0]
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

def generate_output_flux_img2img(image: Image.Image, low_threshold: int, high_threshold: int, prompt: str, width: int, height: int, steps: int):
    try:
        processor = CannyDetector()
        control_image = processor(image, low_threshold=low_threshold, high_threshold=high_threshold, detect_resolution=1024, image_resolution=1024, output_type="pil")
        
        result = manager.get_model('flux_img2img')(
            prompt,
            control_image=image,
            height=height,
            width=width,
            guidance_scale=30.0,
            output_type="pil",
            num_inference_steps=steps,
            max_sequence_length=512,
        )
        return result.images[0]
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

def prepare_img(image: Image.Image, low_threshold: int, high_threshold: int, output_type: str = "pil"):
    try:
        processor = CannyDetector()
        control_image = processor(image, low_threshold=low_threshold, high_threshold=high_threshold, detect_resolution=1024, image_resolution=1024, output_type="pil")

        return control_image
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")