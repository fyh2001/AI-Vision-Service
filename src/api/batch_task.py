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
TEXT2IMAGE_BATCH_SIZE = int(os.getenv("TEXT2IMAGE_BATCH_SIZE", 2))
IMAGE2TEXT_BATCH_SIZE = int(os.getenv("IMAGE2TEXT_BATCH_SIZE", 2))
BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", 0.3))

queue = asyncio.Queue(maxsize=1)

model = None


async def batch_worker():
    print(
        f"[Worker] Started with batch size {TEXT2IMAGE_BATCH_SIZE} and timeout {BATCH_TIMEOUT}"
    )

    while True:
        first_task = await queue.get()
        batch = [first_task]
        start_time = time()

        batch_size = (
            TEXT2IMAGE_BATCH_SIZE
            if MODEL_TYPE == "text2image"
            else IMAGE2TEXT_BATCH_SIZE
        )
        while len(batch) < batch_size and (time() - start_time) < BATCH_TIMEOUT:
            try:
                task = queue.get_nowait()
                batch.append(task)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)

        funcs, inputs, futs = zip(*batch)
        try:
            results = await asyncio.to_thread(funcs[0], inputs)

            if not isinstance(results, list):
                results = [results]
            for fut, result in zip(futs, results):
                fut.set_result(result)
        except Exception as e:
            for fut in futs:
                fut.set_exception(e)
        finally:
            for task in batch:
                queue.task_done()


async def enqueue_task(func, input_data):
    fut = asyncio.get_event_loop().create_future()
    await queue.put((func, input_data, fut))
    return await fut


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = get_model(MODEL_TYPE)
    print(f"Model loaded: {model}")
    asyncio.create_task(batch_worker())
    yield


app = FastAPI(title="AI Vision Batch Service", lifespan=lifespan)


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
