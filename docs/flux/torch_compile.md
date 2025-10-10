# Torch Compile

## 使用 torch compile 加速推理:

```Python
import torch
from diffusers import FluxPipeline
from torch.profiler import profile, record_function, ProfilerActivity
from src.models.base import BaseModel

class FluxModel(BaseModel):
    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        compile_model: bool = True,
        compile_mode: str = "reduce-overhead",
    ):
        super().__init__()
        self.model = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.to(self.device)

        if compile_model and torch.cuda.is_available():
            self.model.transformer = torch.compile(
                self.model.transformer,
                mode=compile_mode,
                fullgraph=True
            )
```

这里有个细节，我一开始看了一下 `GPT & Doubao` 给我的样例，都是对 `model.unet` 进行 compile。

但是实际执行会报错：

```bash
AttributeError: 'FluxPipeline' object has no attribute 'unet'
```

原因是 `FluxPipeline` 是 **Black Forest Labs 的私有实现** ，内部结构不同。于是我 print 了`self.model`

```Python
FluxPipeline = {
  "_class_name": "FluxPipeline",
  "_diffusers_version": "0.35.1",
  "_name_or_path": "black-forest-labs/FLUX.1-dev",

  ...

  "transformer": [
    "diffusers",
    "FluxTransformer2DModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

显然 `FluxPipeline` 中核心生成网络是 `transformer: FluxTransformer2DModel`, 所以 `transformer`才是我们应该 compile 的对象。

当我执行这个代码，又出现了报错:

```bash
File "/opt/tiger/cuda-test/AI-Vision-Service/.venv/lib/python3.10/site-packages/torch/_inductor/scheduler.py", line 3432, in create_backend
    raise RuntimeError( torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: RuntimeError: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at https://github.com/openai/triton

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information

You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
ERROR: Application startup failed. Exiting.
```

原因是 **TorchInductor** 编译依赖 `Triton`，这里我把 `triton` 和 `setuptools` 都加上就可以了!

```Python
def batch_infer(
    self,
    inputs: list[TextToImageInput],
    height: int = 1024,
    width: int = 1024,
) -> list[TextToImageOutput]:
    prompts = [input.prompt for input in inputs]
    output = self.model(
        prompts,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator(device=self.device),
    )
    return [TextToImageOutput(image=image) for image in output.images]

if __name__ == "__main__":
    prompts = [
        "A beautiful sunset over a calm ocean",
        "A cat sitting on a windowsill",
        "A sunset in the mountains",
        "A city skyline at night",
        "A group of people at a conference",
    ]
    model = FluxModel()
    outputs = model.batch_infer([TextToImageInput(prompt=prompt) for prompt in prompts])
```

但是我在这里等了差不多有 10 分钟吧，感觉实在太久了~

```bash
Loading checkpoint shards: 100%|███████████████████████████████████████████| 2/2 [00:01<00:00,  1.91it/s]
Loading pipeline components...:  71%|██████████████████████████▎           | 5/7 [00:01<00:00,  2.91it/s]
Loading checkpoint shards: 100%|███████████████████████████████████████████| 3/3 [00:02<00:00,  1.34it/s]
Loading pipeline components...: 100%|██████████████████████████████████████| 7/7 [00:03<00:00,  1.76it/s]
  0%|                                                                      | 0/50 [00:00<?, ?it/s]
```

查阅了一下 **PyTorch** 文档，理解除了 `default` 以外的 mode，都或多或少会加重编译。所以这里索性先把默认的 `mode` 改为 `default`

> Can be either “default”, “reduce-overhead”, “max-autotune” or “max-autotune-no-cudagraphs”
>
> - ”default” is the default mode, which is a good balance between performance and overhead
>
> - ”reduce-overhead” is a mode that reduces the overhead of python with CUDA graphs, useful for small batches. Reduction of overhead can come at the cost of more memory usage, as we will cache the workspace memory required for the invocation so that we do not have to reallocate it on subsequent runs. Reduction of overhead is not guaranteed to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs. There are other circumstances where CUDA graphs are not applicable; use TORCH_LOG=perf_hints to debug.
>
> - ”max-autotune” is a mode that leverages Triton or template based matrix multiplications on supported devices and Triton based convolutions on GPU. It enables CUDA graphs by default on GPU.
>
> - ”max-autotune-no-cudagraphs” is a mode similar to “max-autotune” but without CUDA graphs

终于跑出了结果，但是！发现 **compile** 之后耗时更长了！！这不科学啊，官方文档里起码都有 20% ~ 30% 提升。

```bash
# No torch compile
100%|██████████████████████████████████████████████| 50/50 [02:59<00:00,  3.59s/it]
Batch inference time: 183.1027 seconds

# torch compiled
100%|██████████████████████████████████████████████| 50/50 [03:44<00:00,  4.49s/it]
Batch inference time: 229.4142 seconds
```

请教了一波 GPT 老师，了解到 `torch.compile` 是 `懒编译` 的，会在第一次遇到某个“输入特征 + 运行条件”时做一次 compile。所以我们可以在加载 model 时，先做一次 **warmup** ，这样在下一次正式推理时，就不需要再花时间 compile 了。

```Python
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
                self.model.transformer,
                mode=compile_mode,
                fullgraph=True
            )

        if do_warmup:
            with torch.inference_mode():
                print("[FluxModel] Running warmup inference for compile optimization...")
                _ = self.model(
                    "A simple placeholder image",
                    height=256,
                    width=256,
                    guidance_scale=3.5,
                    num_inference_steps=2,
                    max_sequence_length=64,
                    generator=torch.Generator(device=self.device),
                )
                print("[FluxModel] Warmup complete — compiled graph ready.")
```

但是结果还是不对啊！！这 warmup 之后怎么好像并没有提升。

```bash
[FluxModel] Running warmup inference for compile optimization...
100%|██████████████████████████████████████████████| 2/2 [00:31<00:00, 15.88s/it]
[FluxModel] Warmup complete — compiled graph ready.
100%|██████████████████████████████████████████████| 50/50 [04:03<00:00,  4.87s/it]
Batch inference time: 246.9110 seconds
```

我后面尝试在 **warmup** 的时候多算几步：num_inference_steps = 10 | 20 | 30 | 50 但好像都没有效果

于是我尝试把其他参数都和 **正式推理** 时的对齐 **(但依旧没有效果)**:

```Python
if do_warmup:
    with torch.inference_mode():
        print("[FluxModel] Running warmup inference for compile optimization...")
        _ = self.model(
            "A simple placeholder image",
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator(device=self.device),
        )
        print("[FluxModel] Warmup complete — compiled graph ready.")
```

关于这一部份，PyTorch 文档中好像没有特别明确指出要用什么姿势打开 warmup，我结合 GPT 大概理解了一下：

- warmup 阶段应模拟或逼近真实推理阶段的所有关键环境条件与输入特性
- 需要对其以下（包括但不限于）参数/条件：
  - tensor shape
  - dtype
  - batch size
  - 设备 / 加速器 (GPU, CPU) / 流(stream) / 上下文
  - 模型状态 (权重、BatchNorm 状态、dropout 模式, eval/train 模式)
  - 编译参数 / torch.compile 的 mode / options / fullgraph / dynamic

我在 **正式推理** 时的 `batch_size` 为 5， 所以在 `warmup` 阶段的 `batch_size` 也应该为 5！！

```Python
if do_warmup:
    warmup_prompt = [
        "A simple placeholder image",
        "A sunset in the mountains",
        "A city skyline at night",
        "A group of people at a conference",
        "A group of people at a conference",
    ]
    with torch.inference_mode():
        print("[FluxModel] Running warmup inference for compile optimization...")
        _ = self.model(
            warmup_prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator(device=self.device),
        )
        print("[FluxModel] Warmup complete — compiled graph ready.")
```

这一次神功大成！对比 no torch compile 的 case，compile 后效率提升接近 **15%** ！

```bash
[FluxModel] Running warmup inference for compile optimization...
100%|██████████████████████████████████████████████| 50/50 [03:04<00:00,  3.70s/it]
[FluxModel] Warmup complete — compiled graph ready.
100%|██████████████████████████████████████████████| 50/50 [02:31<00:00,  3.02s/it]
Batch inference time: 155.0099 seconds
```

我又对比了一下生成 `512 * 512` 图片的 case，compile 后效率提升接近 **20%** ！

```bash
# No torch compile
100%|██████████████████████████████████████████████| 50/50 [00:52<00:00,  1.05s/it]
Batch inference time: 54.6904 seconds

# torch compiled
100%|██████████████████████████████████████████████| 50/50 [00:42<00:00,  1.19it/s]
Batch inference time: 43.5046 seconds
```

### 更多尝试

对同一个 case 我做了以下尝试，但是都没啥实质性的提升，记录一下。

- 开启 GPU Graphs

  ```Python
  import torch

  torch._inductor.config.triton.cudagraphs = enable_cuda_graph
  ```

- 调整 `compile mode = "reduce-overhead"`

- 将 model 内部的 `vae`, `text_encoder`, `text_encoder_2` 都 compile
  ```Python
  self.model.vae = torch.compile(self.model.vae, mode=compile_mode, fullgraph=True)
  self.model.text_encoder = torch.compile(self.model.text_encoder, mode=compile_mode, fullgraph=True)
  self.model.text_encoder_2 = torch.compile(self.model.text_encoder_2, mode=compile_mode, fullgraph=True)
  ```
- warmup 次数增多至 5 次
  ```Python
  with torch.inference_mode():
      print("[FluxModel] Running warmup inference for compile optimization...")
      for _ in range(5):
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
  ```

## torch compile 与 torch profiler

我想对比一下 **torch no compile** 和 **torch compiled** 的 **profiler** 差异，于是

```Python
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
```

No torch compile:
<img src="../../public/flux_no_torch_compile.png">

torch compiled:
<img src="../../public/flux_torch_compiled.png">

简单观察可以发现，**compile** 前的有不少多个连续的 **小 Kernel**，而 **compile** 后，这些 **小 Kernel** 被融合成了少数几个大的 **`复合 Kernel`**。可以见得，这就是 `torch.compile` 对算子的核心优化。

看了一些资料，结合我个人的理解，**compile** 能够加速推理主要是因为：

- 每个 `Kernel` 启动都有**固定开销**，比如 CPU 向 GPU 传递参数、GPU 调度线程块等操作，这些开销都会 "分摊"到每个小算子上，导致整体利用率低。

- 在 `No torch compile` 图中，前半段有一大片花花绿绿的**短小密集**的`Kernel`，那么在执行的时候，GPU 会在 "启动 Kernel -> 计算 -> 等待下一个 Kernel" 之间频繁切换，这样空闲时间占比就高了。

- `torch.compile` 会静态分析计算图，会将"可以连续执行、数据依赖紧密"的多个小 Kernel 融合成一个复合 Kernel，然后编译成一个大的 GPU Kernel。

  - 那原本需要 10 次 Kernel 启动的操作，融合后可能就只需要 1 次，省去了 9 次启动的固定开销！

  - 而且融合后的大 Kernel 能够让 GPU 计算单元"连续工作"（我理解是一些依赖中间结果的操作，会将中间结果在寄存器内直接传递，不再需要读写全局内存）

  - 所以在 `torch compiled` 图中，不再有像`No torch compile`图里一大片连续的花花绿绿的 Kernel 了，看起来也舒服很多！
