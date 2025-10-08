import asyncio
import base64
import io
import os
from contextlib import asynccontextmanager
from time import time

import uvicorn
from fastapi import FastAPI, Request

from model_registry import get_model
from models.flux_1 import TextToImageInput
from models.qwen2_5_vl import ImageToTextInput

MODEL_TYPE = os.getenv("MODEL_TYPE", "text2image").lower()

queue = asyncio.Queue(maxsize=1)

model = None


async def worker_loop():
    while True:
        func, args, kwargs, fut = await queue.get()
        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
            fut.set_result(result)
        except Exception as e:
            fut.set_exception(e)
        finally:
            queue.task_done()


async def enqueue_task(func, *args, **kwargs):
    fut = asyncio.get_event_loop().create_future()
    await queue.put((func, args, kwargs, fut))
    return await fut


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = get_model(MODEL_TYPE)
    print(f"Model loaded: {model}")
    asyncio.create_task(worker_loop())
    yield


app = FastAPI(title="AI Vision Service", lifespan=lifespan)


@app.post("/infer")
async def infer(request: Request):
    input_data = await request.json()
    start_time = time()
    if MODEL_TYPE == "text2image":
        input_data = TextToImageInput(prompt=input_data.get("prompt"))
        output_data = await enqueue_task(model.infer, input_data)

        end_time = time()
        print(f"Text2Image request done in {end_time - start_time:.2f}s")

        buffer = io.BytesIO()
        output_data.image.save(buffer, format="PNG")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image_base64": img_b64}
    elif MODEL_TYPE == "image2text":
        input_data = ImageToTextInput(
            images=input_data.get("images"), prompt=input_data.get("prompt")
        )
        output_data = await enqueue_task(model.infer, input_data)

        end_time = time()
        print(f"Image2Text request done in {end_time - start_time:.2f}s")
        return {"text": output_data.text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
