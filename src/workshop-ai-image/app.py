import gradio as gr
from PIL import Image
from inference import generate_output_flux_txt2img, generate_output_flux_img2img, prepare_img


def generate_text2img(prompt: str, width: int, height: int, steps: int):
    return generate_output_flux_txt2img(prompt, width, height, steps)
    
def generate_img2img(image: Image.Image, low_threshold: int, high_threshold: int, prompt: str, width: int, height: int, steps: int):
    return generate_output_flux_img2img(image, low_threshold, high_threshold, prompt, width, height, steps)

def generate_prepare_img(image: Image.Image, low_threshold: int, high_threshold: int):
    return prepare_img(image, low_threshold, high_threshold)

# INTERFACE DEFINITION
with gr.Blocks() as iface:
    with gr.Tab("📄➡️🖼️ Text2Image"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    prompt = gr.Textbox(
                        label="✍🏼 Prompt",
                        info="Enter your prompt to create an image based on your description.",
                        lines=5,
                        placeholder="Enter your prompt",
                    )
                    generate_btn = gr.Button(
                        value="Submit",
                        variant="primary",
                        scale=2,
                    )
                    
                with gr.Accordion("🖼️ Output Settings", open=False):
                    with gr.Row():
                        width = gr.Number(
                            label="Width",
                            info="The width in pixels of the generated image. This is set to 1024 by default for the best results.",
                            minimum=256,
                            maximum=2048,
                            value=1024,
                        )
                        height = gr.Number(
                            label="Height",
                            info="The height in pixels of the generated image. This is set to 1024 by default for the best results.",
                            minimum=256,
                            maximum=2048,
                            value=1024,
                        )
                    with gr.Row():
                        steps = gr.Number(
                            label="Inference Steps",
                            info="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
                            minimum=8,
                            maximum=50,
                            value=28,
                        )
    
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Generated Output")
    
        # EVENT HANDLING
        generate_btn.click(fn=generate_text2img, inputs=[prompt, width, height, steps], outputs=[image])
    with gr.Tab("🖼️➡️🖼️ Image2Image"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    image_input = gr.Image(type="pil", label="Input image")
                    prompt = gr.Textbox(
                        label="✍🏼 Prompt",
                        info="Enter your prompt to create an image based on your description.",
                        lines=5,
                        placeholder="Enter your prompt",
                    )
                    generate_btn = gr.Button(
                        value="Submit",
                        variant="primary",
                        scale=2,
                    )
                    generate_prep_btn = gr.Button(
                        value="Prepare",
                        variant="primary",
                        scale=2,
                    )
                with gr.Accordion("🖼️ Prepare Settings", open=True):
                    with gr.Row():
                        low_threshold = gr.Number(
                            label="Low Threshold",
                            minimum=0,
                            maximum=255,
                            value=50,
                        )
                        high_threshold = gr.Number(
                            label="High Threshold",
                            minimum=0,
                            maximum=255,
                            value=100,
                        )
                    
                with gr.Accordion("🖼️ Output Settings", open=False):
                    with gr.Row():
                        width = gr.Number(
                            label="Width",
                            info="The width in pixels of the generated image. This is set to 1024 by default for the best results.",
                            minimum=256,
                            maximum=2048,
                            value=1024,
                        )
                        height = gr.Number(
                            label="Height",
                            info="The height in pixels of the generated image. This is set to 1024 by default for the best results.",
                            minimum=256,
                            maximum=2048,
                            value=1024,
                        )
                    with gr.Row():
                        steps = gr.Number(
                            label="Inference Steps",
                            info="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
                            minimum=8,
                            maximum=50,
                            value=28,
                        )

            with gr.Column(scale=1):
                image_prep = gr.Image(type="pil", label="Prepare Output")
                image_out = gr.Image(type="pil", label="Generated Output")
    
        # EVENT HANDLING
        generate_btn.click(fn=generate_img2img, inputs=[image_input, low_threshold, high_threshold, prompt, width, height, steps], outputs=[image_out])
        generate_prep_btn.click(fn=prepare_img, inputs=[image_input, low_threshold, high_threshold], outputs=[image_prep])

if __name__ == "__main__":
    iface.launch(server_name='0.0.0.0', server_port=7860)