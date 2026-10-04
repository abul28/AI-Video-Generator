import tempfile
import torch
import gradio as gr
import spaces

from diffusers import WanPipeline
from diffusers.utils import export_to_video


MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

print("Loading Wan2.1...")

pipe = WanPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()

print("Model loaded.")


@spaces.GPU(duration=120)
def generate_video(prompt):

    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    output = pipe(
        prompt=prompt,
        negative_prompt=(
            "blurry, low quality, distorted, "
            "static image, text, watermark"
        ),
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0
    )

    frames = output.frames[0]

    output_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name

    export_to_video(
        frames,
        output_file,
        fps=16
    )

    return output_file


with gr.Blocks(
    title="AI Video Maker"
) as demo:

    gr.Markdown(
        """
        # 🎬 AI Video Maker

        Generate short AI videos from text.
        """
    )

    prompt = gr.Textbox(
        label="Video Prompt",
        placeholder=(
            "Example: Cinematic Earth rotating in space..."
        ),
        lines=5
    )

    generate = gr.Button(
        "🎬 Generate Video",
        variant="primary"
    )

    video = gr.Video(
        label="Generated Video"
    )

    generate.click(
        fn=generate_video,
        inputs=prompt,
        outputs=video
    )


demo.launch()