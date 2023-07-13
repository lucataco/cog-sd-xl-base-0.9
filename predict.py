from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline
import tempfile
import torch
import gc

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-0.9"
MODEL_CACHE = "cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=MODEL_CACHE,
        )
        self.pipe.to("cuda")

    # Define the arguments and types the model takes as input
    def predict(
        self,
        prompt: str = "full body,Cyber goth Geisha in the rain in a  tokyo future  city city wide, Pretty Face, Beautiful eyes",
        negative_prompt: str = "",
        steps: int = Input(description=" # Inference Steps", ge=0, le=100, default=50),
        guidance: float = Input(description="Guidance scale", default=9),
    ) -> Path:
        """Run a single prediction on the model"""
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]

        gc.collect()
        torch.cuda.empty_cache()

        output_path = Path(tempfile.mkdtemp()) / "output.png"
        image.save(output_path)

        return  Path(output_path)
