import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from dataclasses import dataclass
from src.models.base import BaseModel
from torch.profiler import profile, ProfilerActivity

import time
import contextlib


@dataclass
class ImageToTextInput:
    images: list[str]
    prompt: str


@dataclass
class ImageToTextOutput:
    text: str


class Qwen2_5_VLModel(BaseModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        compile_model: bool = True,
        compile_mode: str = "default",
        do_warmup: bool = True,
    ):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model.to(self.device)

        if compile_model and torch.cuda.is_available():
            self.model = torch.compile(self.model, mode=compile_mode, fullgraph=True)

            if do_warmup:
                warmup_image_inputs = [
                    "outputs/image_0.png",
                    "outputs/image_1.png",
                    "outputs/image_2.png",
                ]
                warmup_messages = [
                    self.template_input("Describe the image", [image])
                    for image in warmup_image_inputs
                ]
                warmup_batch_text = [
                    self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    for messages in warmup_messages
                ]
                batch_image_inputs, warmup_video_inputs = process_vision_info(
                    warmup_messages
                )
                batch_inputs = self.processor(
                    text=warmup_batch_text,
                    images=batch_image_inputs,
                    videos=warmup_video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                print(
                    "[Qwen2_5_VLModel] Running warmup inference for compile optimization..."
                )
                with torch.inference_mode():
                    _ = self.model.generate(
                        **batch_inputs,
                        max_new_tokens=128,
                    )
                print("[Qwen2_5_VLModel] Warmup complete — compiled graph ready.")

    def template_input(self, prompt: str, images: list[str]) -> list:
        messages = [{"role": "user", "content": []}]

        for image in images:
            messages[0]["content"].append({"type": "image", "image": image})
        messages[0]["content"].append({"type": "text", "text": prompt})

        return messages

    @torch.inference_mode()
    def infer(
        self,
        input: ImageToTextInput,
        max_new_tokens: int = 128,
    ) -> ImageToTextOutput:
        messages = self.template_input(input.prompt, input.images)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return ImageToTextOutput(text=output_text[0])

    @torch.inference_mode()
    def batch_infer(
        self,
        inputs: list[ImageToTextInput],
        max_new_tokens: int = 128,
        enable_profiler: bool = False,
        profiler_output_file: str = "qwen2_5_vl_trace.json",
    ) -> list[ImageToTextOutput]:
        if enable_profiler:
            prof_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=True,
                with_stack=True,
            )
        else:
            prof_context = contextlib.nullcontext()
        time_start = time.time()
        with prof_context as profiler:
            batch_messages = [
                self.template_input(input.prompt, input.images) for input in inputs
            ]
            batch_text = [
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in batch_messages
            ]
            batch_image_inputs, batch_video_inputs = process_vision_info(batch_messages)

            batch_inputs = self.processor(
                text=batch_text,
                images=batch_image_inputs,
                videos=batch_video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            generated_ids = self.model.generate(
                **batch_inputs, max_new_tokens=max_new_tokens
            )

            generated_ids_trimmed = [
                ids[len(in_ids) :]
                for in_ids, ids in zip(batch_inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        if enable_profiler:
            profiler.export_chrome_trace(profiler_output_file)
        print(f"batch_infer time: {time.time() - time_start}")

        return [ImageToTextOutput(text=text) for text in output_text]


if __name__ == "__main__":
    images = [
        "outputs/image_0.png",
        "outputs/image_1.png",
        "outputs/image_3.png",
    ]

    model = Qwen2_5_VLModel(compile_model=True)
    # output = model.infer(
    #     ImageToTextInput(images=images, prompt="Describe the image")
    # )
    outputs = model.batch_infer(
        [
            ImageToTextInput(images=[image], prompt="Describe the image")
            for image in images
        ],
        enable_profiler=False,
        profiler_output_file="qwen2_5_vl_trace_with_compile_5.json",
    )

    for output in outputs:
        print(output.text[:60])
