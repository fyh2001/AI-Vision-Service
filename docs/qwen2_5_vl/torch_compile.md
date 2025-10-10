# Torch Compile

## 使用 torch compile 加速推理:

因为已经有了 `Flux` 的 `torch.compile` 经验，再对 `Qwen2.5_vl` 做 `torch.compile` 优化明显上手快很多

```Python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.models.base import BaseModel

class Qwen2_5_VLModel(BaseModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        compile_model: bool = True,
        compile_mode: str = "default",
        do_warmup: bool = True,
        batch_size: int = 3,
    ):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(self.device)

        if compile_model and torch.cuda.is_available():
            self.model = torch.compile(
                self.model, mode=compile_mode, fullgraph=True
            )

            if do_warmup:
                warmup_image_inputs = [f"outputs/image_{i}.png" for i in range batch_size]
                warmup_messages = [self.template_input("Describe the image", [image]) for image in warmup_image_inputs]
                warmup_batch_text = [
                    self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    for messages in warmup_messages
                ]
                batch_image_inputs, warmup_video_inputs = process_vision_info(warmup_messages)
                batch_inputs = self.processor(
                    text=warmup_batch_text,
                    images=batch_image_inputs,
                    videos=warmup_video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                print("[Qwen2_5_VLModel] Running warmup inference for compile optimization...")
                with torch.inference_mode():
                    _ = self.model.generate(
                        **batch_inputs,
                        max_new_tokens=128,
                    )
                print("[Qwen2_5_VLModel] Warmup complete — compiled graph ready.")
```

效率提升接近 10%，还是很明显的。

```bash
# No torch.compile
batch_infer time: 10.121410608291626

# torch.compile
batch_infer time: 9.056618452072144
```

这里我还了解到一个新东西，**model.processor** 中的 `use_fast`。

大致就是 `use_fast=True` 时，使用的是**支持多线程 SIMD 优化** 的 `FastTokenizer`，在批量 tokenization 时比纯 Python 版本(use_fast=False)的 tokenizer 更快。

```python
self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

# python3 test_qwen_2_5_vl.py
batch_infer time: 8.901242733001709
```
