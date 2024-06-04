import os
from text_summarizer.logging.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from text_summarizer.entity.configuration import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizers = {
            'pegasus': AutoTokenizer.from_pretrained(config.tokenizer_names[0]),
            'bart': AutoTokenizer.from_pretrained(config.tokenizer_names[1])
        }

    def convert_examples_to_features(self, example_batch, tokenizer):
        input_encodings = tokenizer(example_batch['text'], max_length=1024, truncation=True)
        with tokenizer.as_target_tokenizer():
            target_encodings = tokenizer(example_batch['summary'], max_length=128, truncation=True)
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def convert(self):
        dataset = load_from_disk(self.config.data_path)
        for name, tokenizer in self.tokenizers.items():
            dataset_pt = dataset.map(lambda x: self.convert_examples_to_features(x, tokenizer), batched=True)
            dataset_pt.save_to_disk(os.path.join(self.config.root_dir, f"{name}_dataset"))