import asyncio
import base64
import aiohttp
import os
from time import time


HTTP_URL = "http://localhost:8000/infer"
WS_URL = "ws://localhost:8000/ws"

MODE = "text2image"  # 可选: "text2image" 或 "image2text"
OUTPUT_DIR = "outputs"
NUM_REQUESTS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


async def test_http_text2image(session: aiohttp.ClientSession, prompt: str, idx: int):
    """测试文生图接口"""
    payload = {"prompt": prompt}
    start_time = time()

    async with session.post(HTTP_URL, json=payload) as resp:
        end_time = time()
        if resp.status != 200:
            print(f"[HTTP {idx}] ❌ Error {resp.status}: {await resp.text()}")
            return
        data = await resp.json()
        img_base64 = data.get("image_base64")

        if not img_base64:
            print(f"[HTTP {idx}] ⚠️ No image data returned")
            return

        img_bytes = base64.b64decode(img_base64)
        img_path = os.path.join(OUTPUT_DIR, f"http_output_{idx}.png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        print(f"[HTTP {idx}] ✅ Done in {end_time - start_time:.2f}s, saved {img_path}")


async def test_http_image2text(
    session: aiohttp.ClientSession, img_path: str, prompt: str, idx: int
):
    """测试图生文接口"""
    # with open(img_path, "rb") as f:
    #     img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {"images": [img_path], "prompt": prompt}
    start_time = time()

    async with session.post(HTTP_URL, json=payload) as resp:
        end_time = time()
        if resp.status != 200:
            print(f"[HTTP {idx}] ❌ Error {resp.status}: {await resp.text()}")
            return
        data = await resp.json()
        text = data.get("text", "")
        print(
            f"[HTTP {idx}] ✅ {end_time - start_time:.2f}s | Response: {text[:60]}..."
        )


async def main(prompts: list[str]):
    timeout = aiohttp.ClientTimeout(
        total=600,  # 总超时 10 分钟
        connect=10,  # 连接超时 10s
        sock_connect=10,  # 套接字连接超时 10s
        sock_read=600,  # 读取超时 10 分钟
    )
    if MODE == "text2image":
        async with aiohttp.ClientSession(timeout=timeout) as session:
            print("🚀 Testing HTTP Text2Image...")
            tasks = [
                test_http_text2image(session, prompts[i], i + 1)
                for i in range(NUM_REQUESTS)
            ]
            await asyncio.gather(*tasks)
    elif MODE == "image2text":
        async with aiohttp.ClientSession(timeout=timeout) as session:
            print("🚀 Testing HTTP Image2Text...")
            tasks = [
                test_http_image2text(
                    session,
                    os.path.join(OUTPUT_DIR, f"http_output_{i + 1}.png"),
                    "Describe this image",
                    i + 1,
                )
                for i in range(NUM_REQUESTS)
            ]
            await asyncio.gather(*tasks)
    else:
        print("❌ Unknown MODE:", MODE)


if __name__ == "__main__":
    prompts = [
        "A beautiful sunset over a calm ocean",
        "A cat sitting on a windowsill",
        "A sunset in the mountains",
        "A city skyline at night",
        "A group of people at a conference",
    ]
    asyncio.run(main(prompts))
