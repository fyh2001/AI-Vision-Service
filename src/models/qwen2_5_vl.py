import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from dataclasses import dataclass
from src.models.base import BaseModel


@dataclass
class ImageToTextInput:
    images: list[str]
    prompt: str


@dataclass
class ImageToTextOutput:
    text: str


class Qwen2_5_VLModel(BaseModel):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(self.device)

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

    def batch_infer(
        self,
        inputs: list[ImageToTextInput],
        max_new_tokens: int = 128,
    ) -> list[ImageToTextOutput]:
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
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(batch_inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return [ImageToTextOutput(text=text) for text in output_text]


if __name__ == "__main__":
    model = Qwen2_5_VLModel()
    output = model.infer(
        ImageToTextInput(images=["image.png"], prompt="Describe the image")
    )

    print(output.text)
