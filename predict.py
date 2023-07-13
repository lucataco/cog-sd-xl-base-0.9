from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline
import tempfile
import torch
import gc
import os

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
        # pytorch 2 optimization
        self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

    # Define the arguments and types the model takes as input
    def predict(
        self,
        prompt: str = "highly detailed portrait of an underwater city, with towering spires and domes rising up from the ocean",
        negative_prompt: str = "",
        steps: int = Input(description=" # Inference Steps", ge=0, le=100, default=50),
        guidance: float = Input(description="Guidance scale", default=9),
        seed: int = Input(description="Seed (0 = random, maximum: 2147483647)", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed == 0:
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]

        gc.collect()
        torch.cuda.empty_cache()

        output_path = Path(tempfile.mkdtemp()) / "output.png"
        image.save(output_path)

        return  Path(output_path)
