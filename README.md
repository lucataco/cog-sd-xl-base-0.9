# stabilityai/stable-diffusion-xl-base-0.9 Cog model

This is an implementation of the model [stabilityai/stable-diffusion-xl-base-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights <HF_TOKEN>

Then, you can run predictions:

    cog predict -i prompt="highly detailed portrait of an underwater city, with towering spires and domes rising up from the ocean"
