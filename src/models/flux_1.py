import torch

from dataclasses import dataclass
from diffusers import FluxPipeline
from src.models.base import BaseModel
from PIL import Image


@dataclass
class TextToImageInput:
    prompt: str


@dataclass
class TextToImageOutput:
    image: Image.Image


class FluxModel(BaseModel):
    def __init__(self, model_name: str = "black-forest-labs/FLUX.1-dev"):
        super().__init__()
        self.model = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.to(self.device)

    @torch.inference_mode()
    def infer(
        self,
        input: TextToImageInput,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
    ) -> TextToImageOutput:
        output = self.model(
            input.prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator(device=self.device),
        )
        return TextToImageOutput(image=output.images[0])

    @torch.inference_mode()
    def batch_infer(
        self,
        inputs: list[TextToImageInput],
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
    ) -> list[TextToImageOutput]:
        prompts = [input.prompt for input in inputs]
        outputs = self.model(
            prompts,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator(device=self.device),
        )
        return [TextToImageOutput(image=output.images[i]) for i in range(len(outputs))]


if __name__ == "__main__":
    prompt = "A beautiful sunset over a calm ocean"

    model = FluxModel()
    output = model.infer(TextToImageInput(prompt=prompt))
    output.image.save("image.png")
