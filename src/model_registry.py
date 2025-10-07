from models.base import BaseModel
from models.flux_1 import FluxModel
from models.qwen2_5_vl import Qwen2_5_VLModel

MODEL_REGISTRY = {
    "text2image": FluxModel,
    "image2text": Qwen2_5_VLModel,
}


def get_model(model_type: str) -> BaseModel:
    cls = MODEL_REGISTRY.get(model_type)
    if not cls:
        raise ValueError(f"Unsupported model type: {model_type}")
    return cls()
