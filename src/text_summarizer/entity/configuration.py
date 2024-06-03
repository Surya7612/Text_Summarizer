from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    tokenizer_name: str
    model_max_length: int
