import torch
from torch.profiler import profile, record_function, ProfilerActivity
from dataclasses import dataclass
from diffusers import FluxPipeline
from src.models.base import BaseModel
from PIL import Image
from time import time
import contextlib


@dataclass
class TextToImageInput:
    prompt: str


@dataclass
class TextToImageOutput:
    image: Image.Image


class FluxModel(BaseModel):
    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        compile_model: bool = True,
        compile_mode: str = "default",
        do_warmup: bool = True,
    ):
        super().__init__()
        self.model = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.to(self.device)

        if compile_model and torch.cuda.is_available():
            self.model.transformer = torch.compile(
                self.model.transformer, mode=compile_mode, fullgraph=True
            )

        if do_warmup:
            warmup_prompt = [
                "A simple placeholder image",
                "A sunset in the mountains",
                "A city skyline at night",
                "A group of people at a conference",
                "A group of people at a conference",
            ]
            with torch.inference_mode():
                print(
                    "[FluxModel] Running warmup inference for compile optimization..."
                )
                _ = self.model(
                    warmup_prompt,
                    height=512,
                    width=512,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    generator=torch.Generator(device=self.device),
                )
                print("[FluxModel] Warmup complete — compiled graph ready.")

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
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
        enable_profiler: bool = False,
    ) -> list[TextToImageOutput]:
        if enable_profiler:
            prof_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=True,
                with_stack=True,
            )
        else:
            prof_context = contextlib.nullcontext()

        with prof_context as profiler:
            prompts = [input.prompt for input in inputs]
            start_time = time()
            output = self.model(
                prompts,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator(device=self.device),
            )
            end_time = time()
            print(f"Batch inference time: {end_time - start_time:.4f} seconds")

        if enable_profiler:
            profiler.export_chrome_trace("perfetto_trace.json")

        return [TextToImageOutput(image=image) for image in output.images]


if __name__ == "__main__":
    prompts = [
        "A beautiful sunset over a calm ocean",
        "A cat sitting on a windowsill",
        "A sunset in the mountains",
        "A city skyline at night",
        "A group of people at a conference",
    ]

    model = FluxModel(do_warmup=False, compile_model=False)
    # output = model.infer(TextToImageInput(prompt=prompts[0]))
    outputs = model.batch_infer(
        [TextToImageInput(prompt=prompt) for prompt in prompts],
        enable_profiler=False,
        height=512,
        width=512,
    )
    for i, output in enumerate(outputs):
        output.image.save(f"outputs/image_{i}.png")
