import os
import torch
import gradio as gr
from diffusers.utils import export_to_video
from diffusers import AutoencoderKL
from contentv_transformer import SD3Transformer3DModel
from contentv_pipeline import ContentVPipeline

# --- NPU Setup (Keep as is if you need it) ---
USE_ASCEND_NPU = int(os.getenv('USE_ASCEND_NPU', '0'))
if USE_ASCEND_NPU:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False
# ---------------------------------------------

# --- Model Loading (Moved outside the function for efficiency) ---
print("Loading models... This may take a moment.")
model_id = 'ByteDance/ContentV-8B'
try:
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
except Exception as e:
    print(f"Could not load VAE with AutoencoderKL, trying AutoencoderKLWan: {e}")
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)


transformer = SD3Transformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = ContentVPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda") # Ensure your device is 'cuda' or 'npu' based on your setup
print("Models loaded successfully!")
# -----------------------------------------------------------------


def generate_video_gradio(prompt: str, negative_prompt: str, num_frames: int, seed: int, progress=gr.Progress()) -> str:
    """
    Generates a video based on the given prompts, number of frames, and a seed,
    with progress updates to the Gradio UI.

    Args:
        prompt (str): The text prompt for video generation.
        negative_prompt (str): The negative prompt to guide video generation away from certain concepts.
        num_frames (int): The number of frames in the output video.
        seed (int): The random seed for reproducibility.
        progress (gr.Progress): Gradio's progress object for UI updates.

    Returns:
        str: The file path to the generated video.
    """
    print(f"Generating video with prompt: '{prompt}', negative prompt: '{negative_prompt}', frames: {num_frames}, seed: {seed}")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Define the callback function to update Gradio's progress bar
    # Adjusted the signature based on contentv_pipeline.py line 381
    def internal_diffusers_callback(pipeline_instance, current_step: int, timestep: int, callback_kwargs):
        # `current_step` is the actual step value we need for progress
        # `pipeline_instance` is the `self` from the pipeline, we don't directly use it for progress here.
        
        # Based on your previous console output (e.g., "4/50"),
        # we'll assume a total of 50 diffusion steps for the primary progress.
        total_diffusion_steps = 50 
        
        if current_step is not None and isinstance(current_step, int):
            current_progress_percentage = (current_step + 1) / total_diffusion_steps
            progress(current_progress_percentage, desc=f"Generating video... (Step {current_step+1}/{total_diffusion_steps})")
        else:
            # This case should ideally no longer be hit if the pipeline consistently passes integers for the second arg
            print(f"Warning: internal_diffusers_callback received invalid 'current_step' value: {current_step}. Skipping progress update.")
        
        # Return an empty dictionary to satisfy ContentVPipeline's expectation
        return {}


    video_frames = pipe(
        num_frames=num_frames,
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        callback_on_step_end=internal_diffusers_callback,
    ).frames[0]

    # Ensure the progress bar completes to 100% after video generation
    progress(1, desc="Video generation complete!")
    
    output_path = "generated_video.mp4"
    export_to_video(video_frames, output_path, fps=24)
    print(f"Video saved to {output_path}")
    return output_path

# --- Gradio Interface ---
iface = gr.Interface(
    fn=generate_video_gradio,
    inputs=[
        gr.Textbox(
            label="Prompt",
            lines=3,
            value="Premium smartwatch rotates elegantly on a glass surface in a minimalist studio. 360-degree camera movement revealing all angles. Clean, bright lighting with subtle reflections. Color palette: all white, monochrome tones, Nikon D850 DSLR 200mm f/1.8 lens, f/2.2 aperture,3D, UHD",
        ),
        gr.Textbox(
            label="Negative Prompt",
            lines=2,
            value="overexposed, low quality, deformation, text, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
        ),
        gr.Slider(
            minimum=50,
            maximum=200,
            step=5,
            value=125,
            label="Number of Frames"
        ),
        gr.Number(
            label="Seed (for reproducibility)",
            value=42,
            minimum=0,
            step=1
        )
    ],
    outputs=gr.Video(label="Generated Video"),
    title="ContentV Text to Video Generator",
    description="Generate high-quality videos from text prompts using the ContentV model. Adjust prompts, frame count, and seed for desired results.",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="teal"),
)

if __name__ == "__main__":
    iface.launch()