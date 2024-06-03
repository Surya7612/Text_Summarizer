from pathlib import Path
from text_summarizer.utils.common import read_yaml

def load_config(config_path='config/config.yaml'):
    config = read_yaml(Path(config_path))
    return config
